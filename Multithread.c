// adaptive_threadpool.c - C23 cross-platform implementation of adaptive thread pool for Python 3.14 free-threading
// SPDX-License-Identifier: MIT

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

// Cross-platform threading headers
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <process.h>
#include <psapi.h>
#include <sysinfoapi.h>
#define THREAD_FUNC DWORD WINAPI
#define THREAD_HANDLE HANDLE
#define THREAD_RETURN DWORD
typedef CRITICAL_SECTION thread_lock_t;
typedef CONDITION_VARIABLE thread_cond_t;
#else
#include <pthread.h>
#include <unistd.h>
#define THREAD_FUNC void*
#define THREAD_HANDLE pthread_t
#define THREAD_RETURN void*
typedef pthread_mutex_t thread_lock_t;
typedef pthread_cond_t thread_cond_t;
#endif

// Define timespec for Windows
#if defined(_WIN32) || defined(_WIN64)
struct timespec {
    time_t tv_sec;
    long tv_nsec;
};
#endif

// Platform-specific includes
#if defined(__linux__)
#include <sys/sysinfo.h>
#include <sys/types.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <sys/proc_info.h>
#endif

// Module version
#define MODULE_VERSION "0.0.1"

// Constants for thread pool management
#define DEFAULT_MIN_THREADS 2
#define DEFAULT_MAX_THREADS 32
#define DEFAULT_SOFT_LIMIT 8
#define DEFAULT_MONITOR_INTERVAL_MS 250
#define DEFAULT_SCALE_UP_THRESHOLD 0.75
#define DEFAULT_SCALE_DOWN_THRESHOLD 0.25
#define DEFAULT_CPU_THRESHOLD 0.80
#define TASK_QUEUE_INITIAL_CAPACITY 128
#define MEMORY_POOL_SIZE (1024 * 1024)  // 1MB memory pool
#define MAX_ALLOCATION_ATTEMPTS 3

// Memory pool for better allocation control
typedef struct {
    uint8_t *base;
    size_t size;
    size_t used;
    thread_lock_t lock;
} MemoryPool;

// Scaling policies
typedef enum {
    POLICY_CONSERVATIVE,
    POLICY_BALANCED,
    POLICY_AGGRESSIVE
} ScalingPolicy;

// Task structure with improved memory management
typedef struct Task {
    PyObject *callable;
    PyObject *args;
    PyObject *kwargs;
    uint64_t priority;
    uint64_t submit_time_ns;
    struct Task *next;
    bool is_malloced;
} Task;

// Task queue with priority support and cross-platform synchronization
typedef struct {
    Task *head;
    Task *tail;
    atomic_size_t size;
    atomic_size_t total_submitted;
    atomic_size_t total_completed;
    thread_lock_t lock;
    thread_cond_t not_empty;
    thread_cond_t not_full;
    size_t capacity;
    atomic_bool shutdown;
    MemoryPool *memory_pool;
} TaskQueue;

// Resource metrics
typedef struct {
    atomic_uint_fast64_t cpu_utilization_percent;  // 0-100
    atomic_uint_fast64_t memory_available_mb;
    atomic_uint_fast64_t active_threads;
    atomic_uint_fast64_t idle_threads;
    atomic_uint_fast64_t queue_depth;
    atomic_uint_fast64_t avg_task_duration_ns;
    atomic_uint_fast64_t tasks_per_second;
    struct timespec last_update;
} ResourceMetrics;

// Thread worker context with cross-platform handle
typedef struct {
    size_t thread_id;
    THREAD_HANDLE handle;
    atomic_bool active;
    atomic_bool should_exit;
    atomic_uint_fast64_t tasks_processed;
    struct AdaptiveThreadPool *pool;
} WorkerThread;

// Main thread pool structure
typedef struct AdaptiveThreadPool {
    // Configuration
    size_t min_threads;
    size_t max_threads;
    size_t soft_limit;
    size_t hard_limit;
    ScalingPolicy policy;
    uint32_t monitor_interval_ms;
    double scale_up_threshold;
    double scale_down_threshold;
    double cpu_threshold;
    
    // Runtime state
    WorkerThread **workers;
    atomic_size_t current_thread_count;
    TaskQueue *task_queue;
    ResourceMetrics metrics;
    MemoryPool *memory_pool;
    
    // Monitoring thread
    THREAD_HANDLE monitor_thread;
    atomic_bool monitor_running;
    
    // Synchronization
    thread_lock_t pool_lock;
    atomic_bool shutdown;
    
    // Python object header
    PyObject_HEAD
} AdaptiveThreadPool;

// Forward declarations
static THREAD_RETURN worker_thread_func(void *arg);
static THREAD_RETURN monitor_thread_func(void *arg);
static int scale_thread_pool(AdaptiveThreadPool *pool, int delta);
static void update_resource_metrics(AdaptiveThreadPool *pool);
static uint64_t get_monotonic_time_ns(void);
static MemoryPool *memory_pool_create(size_t size);
static void memory_pool_destroy(MemoryPool *pool);
static void *memory_pool_alloc(MemoryPool *pool, size_t size);

// ============================================================================
// Cross-platform Threading Abstractions
// ============================================================================

// Lock functions
#if defined(_WIN32) || defined(_WIN64)
static int thread_lock_init(thread_lock_t *lock) {
    InitializeCriticalSection(lock);
    return 0;
}

static int thread_lock_destroy(thread_lock_t *lock) {
    DeleteCriticalSection(lock);
    return 0;
}

static int thread_lock_lock(thread_lock_t *lock) {
    EnterCriticalSection(lock);
    return 0;
}

static int thread_lock_unlock(thread_lock_t *lock) {
    LeaveCriticalSection(lock);
    return 0;
}
#else
static int thread_lock_init(thread_lock_t *lock) {
    return pthread_mutex_init(lock, NULL);
}

static int thread_lock_destroy(thread_lock_t *lock) {
    return pthread_mutex_destroy(lock);
}

static int thread_lock_lock(thread_lock_t *lock) {
    return pthread_mutex_lock(lock);
}

static int thread_lock_unlock(thread_lock_t *lock) {
    return pthread_mutex_unlock(lock);
}
#endif

// Condition variable functions
#if defined(_WIN32) || defined(_WIN64)
static int thread_cond_init(thread_cond_t *cond) {
    InitializeConditionVariable(cond);
    return 0;
}

static int thread_cond_destroy(thread_cond_t *cond) {
    return 0;  // Condition variables don't need destruction on Windows
}

static int thread_cond_wait(thread_cond_t *cond, thread_lock_t *lock) {
    return SleepConditionVariableCS(cond, lock, INFINITE) ? 0 : -1;
}

static int thread_cond_signal(thread_cond_t *cond) {
    WakeConditionVariable(cond);
    return 0;
}

static int thread_cond_broadcast(thread_cond_t *cond) {
    WakeAllConditionVariable(cond);
    return 0;
}
#else
static int thread_cond_init(thread_cond_t *cond) {
    return pthread_cond_init(cond, NULL);
}

static int thread_cond_destroy(thread_cond_t *cond) {
    return pthread_cond_destroy(cond);
}

static int thread_cond_wait(thread_cond_t *cond, thread_lock_t *lock) {
    return pthread_cond_wait(cond, lock);
}

static int thread_cond_signal(thread_cond_t *cond) {
    return pthread_cond_signal(cond);
}

static int thread_cond_broadcast(thread_cond_t *cond) {
    return pthread_cond_broadcast(cond);
}
#endif

// Thread creation and join (mt_ prefix to avoid macOS Mach thread_create symbol conflict)
#if defined(_WIN32) || defined(_WIN64)
static int mt_mt_thread_create(THREAD_HANDLE *handle, THREAD_FUNC func, void *arg) {
    *handle = CreateThread(NULL, 0, func, arg, 0, NULL);
    return *handle != NULL ? 0 : -1;
}

static int mt_mt_thread_join(THREAD_HANDLE handle) {
    return WaitForSingleObject(handle, INFINITE) == WAIT_OBJECT_0 ? 0 : -1;
}

static void thread_sleep_ms(uint32_t ms) {
    Sleep(ms);
}
#else
static int mt_thread_create(THREAD_HANDLE *handle, THREAD_FUNC func, void *arg) {
    return pthread_create(handle, NULL, func, arg);
}

static int mt_thread_join(THREAD_HANDLE handle) {
    return pthread_join(handle, NULL);
}

static void thread_sleep_ms(uint32_t ms) {
    usleep(ms * 1000);
}
#endif

// ============================================================================
// Memory Pool Implementation
// ============================================================================

static MemoryPool *memory_pool_create(size_t size) {
    MemoryPool *pool = malloc(sizeof(MemoryPool));
    if (!pool) {
        return NULL;
    }
    
    pool->base = malloc(size);
    if (!pool->base) {
        free(pool);
        return NULL;
    }
    
    pool->size = size;
    pool->used = 0;
    
    if (thread_lock_init(&pool->lock) != 0) {
        free(pool->base);
        free(pool);
        return NULL;
    }
    
    return pool;
}

static void memory_pool_destroy(MemoryPool *pool) {
    if (!pool) {
        return;
    }
    
    thread_lock_destroy(&pool->lock);
    free(pool->base);
    free(pool);
}

static void *memory_pool_alloc(MemoryPool *pool, size_t size) {
    if (!pool || size == 0) {
        return NULL;
    }
    
    // Align to 8-byte boundary
    size = (size + 7) & ~7;
    
    thread_lock_lock(&pool->lock);
    
    if (pool->used + size > pool->size) {
        thread_lock_unlock(&pool->lock);
        return NULL;  // Pool exhausted
    }
    
    void *ptr = pool->base + pool->used;
    pool->used += size;
    
    thread_lock_unlock(&pool->lock);
    return ptr;
}

// ============================================================================
// Task Queue Implementation
// ============================================================================

static TaskQueue *task_queue_create(size_t capacity) {
    TaskQueue *queue = calloc(1, sizeof(TaskQueue));
    if (!queue) {
        return NULL;
    }
    
    // Create memory pool for task allocations
    queue->memory_pool = memory_pool_create(MEMORY_POOL_SIZE);
    if (!queue->memory_pool) {
        free(queue);
        return NULL;
    }
    
    atomic_init(&queue->size, 0);
    atomic_init(&queue->total_submitted, 0);
    atomic_init(&queue->total_completed, 0);
    atomic_init(&queue->shutdown, false);
    
    if (thread_lock_init(&queue->lock) != 0) {
        memory_pool_destroy(queue->memory_pool);
        free(queue);
        return NULL;
    }
    
    if (thread_cond_init(&queue->not_empty) != 0) {
        thread_lock_destroy(&queue->lock);
        memory_pool_destroy(queue->memory_pool);
        free(queue);
        return NULL;
    }
    
    if (thread_cond_init(&queue->not_full) != 0) {
        thread_cond_destroy(&queue->not_empty);
        thread_lock_destroy(&queue->lock);
        memory_pool_destroy(queue->memory_pool);
        free(queue);
        return NULL;
    }
    
    queue->capacity = capacity;
    queue->head = NULL;
    queue->tail = NULL;
    
    return queue;
}

static void task_queue_destroy(TaskQueue *queue) {
    if (!queue) {
        return;
    }
    
    atomic_store(&queue->shutdown, true);
    thread_cond_broadcast(&queue->not_empty);
    thread_cond_broadcast(&queue->not_full);
    
    // Clean up remaining tasks
    Task *current = queue->head;
    while (current) {
        Task *next = current->next;
        Py_XDECREF(current->callable);
        Py_XDECREF(current->args);
        Py_XDECREF(current->kwargs);
        if (current->is_malloced) {
            free(current);
        }
        current = next;
    }
    
    thread_cond_destroy(&queue->not_empty);
    thread_cond_destroy(&queue->not_full);
    thread_lock_destroy(&queue->lock);
    memory_pool_destroy(queue->memory_pool);
    free(queue);
}

static bool task_queue_enqueue(TaskQueue *queue, PyObject *callable, PyObject *args, PyObject *kwargs, uint64_t priority) {
    if (atomic_load(&queue->shutdown)) {
        return false;
    }
    
    // Try to allocate from memory pool first
    Task *task = memory_pool_alloc(queue->memory_pool, sizeof(Task));
    bool is_malloced = false;
    if (!task) {
        // Fallback to regular allocation
        task = malloc(sizeof(Task));
        if (!task) {
            return false;
        }
        is_malloced = true;
    }
    
    task->callable = callable;
    task->args = args;
    task->kwargs = kwargs;
    task->priority = priority;
    task->submit_time_ns = get_monotonic_time_ns();
    task->next = NULL;
    task->is_malloced = is_malloced;
    
    Py_INCREF(callable);
    Py_XINCREF(args);
    Py_XINCREF(kwargs);
    
    thread_lock_lock(&queue->lock);
    
    // Wait if queue is full
    while (atomic_load(&queue->size) >= queue->capacity && !atomic_load(&queue->shutdown)) {
        thread_cond_wait(&queue->not_full, &queue->lock);
    }
    
    if (atomic_load(&queue->shutdown)) {
        Py_DECREF(callable);
        Py_XDECREF(args);
        Py_XDECREF(kwargs);
        if (is_malloced) {
            free(task);
        }
        thread_lock_unlock(&queue->lock);
        return false;
    }
    
    // Insert by priority (higher priority = lower number, 0 is highest)
    if (!queue->head || priority < queue->head->priority) {
        task->next = queue->head;
        queue->head = task;
        if (!queue->tail) {
            queue->tail = task;
        }
    } else {
        Task *current = queue->head;
        while (current->next && current->next->priority <= priority) {
            current = current->next;
        }
        task->next = current->next;
        current->next = task;
        if (!task->next) {
            queue->tail = task;
        }
    }
    
    atomic_fetch_add(&queue->size, 1);
    atomic_fetch_add(&queue->total_submitted, 1);
    
    thread_cond_signal(&queue->not_empty);
    thread_lock_unlock(&queue->lock);
    
    return true;
}

static Task *task_queue_dequeue(TaskQueue *queue, bool blocking) {
    thread_lock_lock(&queue->lock);
    
    while (atomic_load(&queue->size) == 0 && !atomic_load(&queue->shutdown)) {
        if (!blocking) {
            thread_lock_unlock(&queue->lock);
            return NULL;
        }
        thread_cond_wait(&queue->not_empty, &queue->lock);
    }
    
    if (atomic_load(&queue->shutdown) && atomic_load(&queue->size) == 0) {
        thread_lock_unlock(&queue->lock);
        return NULL;
    }
    
    Task *task = queue->head;
    if (task) {
        queue->head = task->next;
        if (!queue->head) {
            queue->tail = NULL;
        }
        atomic_fetch_sub(&queue->size, 1);
        thread_cond_signal(&queue->not_full);
    }
    
    thread_lock_unlock(&queue->lock);
    return task;
}

// ============================================================================
// Cross-platform Resource Monitoring
// ============================================================================

static uint64_t get_monotonic_time_ns(void) {
#if defined(_WIN32) || defined(_WIN64)
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000000ULL) / frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static double get_cpu_utilization(void) {
#if defined(__linux__)
    static uint64_t prev_total = 0;
    static uint64_t prev_idle = 0;
    
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) {
        return 0.0;
    }
    
    char buffer[256];
    if (!fgets(buffer, sizeof(buffer), fp)) {
        fclose(fp);
        return 0.0;
    }
    fclose(fp);
    
    uint64_t user, nice, system, idle, iowait, irq, softirq;
    sscanf(buffer, "cpu %lu %lu %lu %lu %lu %lu %lu",
           &user, &nice, &system, &idle, &iowait, &irq, &softirq);
    
    uint64_t total = user + nice + system + idle + iowait + irq + softirq;
    uint64_t idle_total = idle + iowait;
    
    if (prev_total == 0) {
        prev_total = total;
        prev_idle = idle_total;
        return 0.0;
    }
    
    uint64_t total_delta = total - prev_total;
    uint64_t idle_delta = idle_total - prev_idle;
    
    prev_total = total;
    prev_idle = idle_total;
    
    if (total_delta == 0) {
        return 0.0;
    }
    
    return ((double)(total_delta - idle_delta) / (double)total_delta) * 100.0;
    
#elif defined(__APPLE__)
    host_cpu_load_info_data_t cpu_info;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                       (host_info_t)&cpu_info, &count) != KERN_SUCCESS) {
        return 0.0;
    }
    
    uint64_t total_ticks = 0;
    for (int i = 0; i < CPU_STATE_MAX; i++) {
        total_ticks += cpu_info.cpu_ticks[i];
    }
    
    static uint64_t prev_total = 0;
    static uint64_t prev_idle = 0;
    
    uint64_t idle = cpu_info.cpu_ticks[CPU_STATE_IDLE];
    
    if (prev_total == 0) {
        prev_total = total_ticks;
        prev_idle = idle;
        return 0.0;
    }
    
    uint64_t total_delta = total_ticks - prev_total;
    uint64_t idle_delta = idle - prev_idle;
    
    prev_total = total_ticks;
    prev_idle = idle;
    
    if (total_delta == 0) {
        return 0.0;
    }
    
    return ((double)(total_delta - idle_delta) / (double)total_delta) * 100.0;
    
#elif defined(_WIN32) || defined(_WIN64)
    static ULARGE_INTEGER prev_idle, prev_kernel, prev_user;
    
    ULARGE_INTEGER idle, kernel, user;
    if (GetSystemTimes((FILETIME*)&idle, (FILETIME*)&kernel, (FILETIME*)&user) == 0) {
        return 0.0;
    }
    
    if (prev_idle.QuadPart == 0) {
        prev_idle = idle;
        prev_kernel = kernel;
        prev_user = user;
        return 0.0;
    }
    
    ULONGLONG idle_delta = idle.QuadPart - prev_idle.QuadPart;
    ULONGLONG kernel_delta = kernel.QuadPart - prev_kernel.QuadPart;
    ULONGLONG user_delta = user.QuadPart - prev_user.QuadPart;
    
    ULONGLONG total_delta = idle_delta + kernel_delta + user_delta;
    
    prev_idle = idle;
    prev_kernel = kernel;
    prev_user = user;
    
    if (total_delta == 0) {
        return 0.0;
    }
    
    return ((double)(total_delta - idle_delta) / (double)total_delta) * 100.0;
#else
    return 0.0;
#endif
}

static uint64_t get_available_memory_mb(void) {
#if defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) != 0) {
        return 0;
    }
    return (si.freeram * si.mem_unit) / (1024 * 1024);
    
#elif defined(__APPLE__)
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                         (host_info64_t)&vm_stats, &count) != KERN_SUCCESS) {
        return 0;
    }
    
    uint64_t free_pages = vm_stats.free_count;
    vm_size_t page_size;
    host_page_size(mach_host_self(), &page_size);

    return (uint64_t)(free_pages * page_size) / (1024 * 1024);
    
#elif defined(_WIN32) || defined(_WIN64)
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status) == 0) {
        return 0;
    }
    return mem_status.ullAvailPhys / (1024 * 1024);
#else
    return 0;
#endif
}

static void update_resource_metrics(AdaptiveThreadPool *pool) {
    double cpu_util = get_cpu_utilization();
    uint64_t mem_avail = get_available_memory_mb();
    size_t queue_depth = atomic_load(&pool->task_queue->size);
    
    atomic_store(&pool->metrics.cpu_utilization_percent, (uint64_t)cpu_util);
    atomic_store(&pool->metrics.memory_available_mb, mem_avail);
    atomic_store(&pool->metrics.queue_depth, queue_depth);
    
    // Calculate active vs idle threads
    size_t active = 0;
    size_t total = atomic_load(&pool->current_thread_count);
    
    for (size_t i = 0; i < total; i++) {
        if (pool->workers[i] && atomic_load(&pool->workers[i]->active)) {
            active++;
        }
    }
    
    atomic_store(&pool->metrics.active_threads, active);
    atomic_store(&pool->metrics.idle_threads, total - active);
    
#if defined(_WIN32) || defined(_WIN64)
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    uint64_t ns = (counter.QuadPart * 1000000000ULL) / frequency.QuadPart;
    pool->metrics.last_update.tv_sec = ns / 1000000000ULL;
    pool->metrics.last_update.tv_nsec = ns % 1000000000ULL;
#else
    clock_gettime(CLOCK_MONOTONIC, &pool->metrics.last_update);
#endif
}

// ============================================================================
// Thread Pool Scaling Logic
// ============================================================================

static int calculate_target_thread_count(AdaptiveThreadPool *pool) {
    size_t current = atomic_load(&pool->current_thread_count);
    size_t queue_depth = atomic_load(&pool->metrics.queue_depth);
    size_t active = atomic_load(&pool->metrics.active_threads);
    size_t idle = atomic_load(&pool->metrics.idle_threads);
    double cpu_util = (double)atomic_load(&pool->metrics.cpu_utilization_percent);
    
    // Don't scale if CPU is saturated
    if (cpu_util > (pool->cpu_threshold * 100.0)) {
        return (int)current;
    }
    
    double load_ratio = (double)active / (double)current;
    
    int target = (int)current;
    
    switch (pool->policy) {
        case POLICY_CONSERVATIVE:
            // Scale up slowly, scale down quickly
            if (load_ratio > pool->scale_up_threshold && queue_depth > current) {
                target = (int)current + 1;
            } else if (load_ratio < pool->scale_down_threshold && idle > 2) {
                target = (int)current - 2;
            }
            break;
            
        case POLICY_BALANCED:
            // Moderate scaling in both directions
            if (load_ratio > pool->scale_up_threshold && queue_depth > current / 2) {
                target = (int)current + ((queue_depth > current * 2) ? 2 : 1);
            } else if (load_ratio < pool->scale_down_threshold && idle > 1) {
                target = (int)current - 1;
            }
            break;
            
        case POLICY_AGGRESSIVE:
            // Scale up quickly, scale down slowly
            if (load_ratio > pool->scale_up_threshold) {
                size_t scale_up = (queue_depth / current) + 1;
                target = (int)current + (scale_up > 4 ? 4 : (int)scale_up);
            } else if (load_ratio < pool->scale_down_threshold && idle > 3) {
                target = (int)current - 1;
            }
            break;
    }
    
    // Apply soft and hard limits
    if (target > (int)pool->hard_limit) {
        target = (int)pool->hard_limit;
    }
    
    if (target > (int)pool->soft_limit) {
        // Allow exceeding soft limit only under high load
        if (queue_depth < pool->soft_limit || cpu_util > (pool->cpu_threshold * 100.0)) {
            target = (int)pool->soft_limit;
        }
    }
    
    if (target < (int)pool->min_threads) {
        target = (int)pool->min_threads;
    }
    
    return target;
}

static int scale_thread_pool(AdaptiveThreadPool *pool, int delta) {
    if (delta == 0) {
        return 0;
    }
    
    thread_lock_lock(&pool->pool_lock);
    
    size_t current = atomic_load(&pool->current_thread_count);
    size_t new_count = current;
    
    if (delta > 0) {
        // Scale up
        for (int i = 0; i < delta && new_count < pool->hard_limit; i++) {
            WorkerThread *worker = calloc(1, sizeof(WorkerThread));
            if (!worker) {
                break;
            }
            
            worker->thread_id = new_count;
            worker->pool = pool;
            atomic_init(&worker->active, false);
            atomic_init(&worker->should_exit, false);
            atomic_init(&worker->tasks_processed, 0);
            
            if (mt_thread_create(&worker->handle, worker_thread_func, worker) != 0) {
                free(worker);
                break;
            }
            
            pool->workers[new_count] = worker;
            new_count++;
        }
    } else {
        // Scale down
        for (int i = 0; i < -delta && new_count > pool->min_threads; i++) {
            size_t idx = new_count - 1;
            WorkerThread *worker = pool->workers[idx];
            if (worker) {
                atomic_store(&worker->should_exit, true);
                thread_lock_unlock(&pool->pool_lock);
                mt_thread_join(worker->handle);
                free(worker);
                thread_lock_lock(&pool->pool_lock);
                pool->workers[idx] = NULL;
                new_count--;
            }
        }
    }
    
    atomic_store(&pool->current_thread_count, new_count);
    thread_lock_unlock(&pool->pool_lock);
    
    return (int)(new_count - current);
}

// ============================================================================
// Worker Thread Implementation
// ============================================================================

#if defined(_WIN32) || defined(_WIN64)
static DWORD WINAPI worker_thread_func(void *arg) {
#else
static void *worker_thread_func(void *arg) {
#endif
    WorkerThread *worker = (WorkerThread *)arg;
    AdaptiveThreadPool *pool = worker->pool;
    
    while (!atomic_load(&pool->shutdown) && !atomic_load(&worker->should_exit)) {
        Task *task = task_queue_dequeue(pool->task_queue, true);
        
        if (!task) {
            continue;
        }
        
        atomic_store(&worker->active, true);
        
        uint64_t start_time = get_monotonic_time_ns();
        
        // Execute the Python callable
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        PyObject *result = PyObject_Call(task->callable, task->args ? task->args : PyTuple_New(0), task->kwargs);
        
        if (!result) {
            PyErr_Print();
        } else {
            Py_DECREF(result);
        }
        
        PyGILState_Release(gstate);
        
        uint64_t end_time = get_monotonic_time_ns();
        uint64_t duration = end_time - start_time;
        
        // Update metrics
        atomic_fetch_add(&pool->task_queue->total_completed, 1);
        atomic_fetch_add(&worker->tasks_processed, 1);
        
        // Update average task duration (simple moving average)
        uint64_t current_avg = atomic_load(&pool->metrics.avg_task_duration_ns);
        uint64_t new_avg = (current_avg * 9 + duration) / 10;
        atomic_store(&pool->metrics.avg_task_duration_ns, new_avg);
        
        // Clean up task
        Py_DECREF(task->callable);
        Py_XDECREF(task->args);
        Py_XDECREF(task->kwargs);
        if (task->is_malloced) {
            free(task);
        }
        
        atomic_store(&worker->active, false);
    }
    
#if defined(_WIN32) || defined(_WIN64)
    return 0;
#else
    return NULL;
#endif
}

// ============================================================================
// Monitor Thread Implementation
// ============================================================================

#if defined(_WIN32) || defined(_WIN64)
static DWORD WINAPI monitor_thread_func(void *arg) {
#else
static void *monitor_thread_func(void *arg) {
#endif
    AdaptiveThreadPool *pool = (AdaptiveThreadPool *)arg;
    
    while (atomic_load(&pool->monitor_running) && !atomic_load(&pool->shutdown)) {
        update_resource_metrics(pool);
        
        int target = calculate_target_thread_count(pool);
        int current = (int)atomic_load(&pool->current_thread_count);
        int delta = target - current;
        
        if (delta != 0) {
            scale_thread_pool(pool, delta);
        }
        
        thread_sleep_ms(pool->monitor_interval_ms);
    }
    
#if defined(_WIN32) || defined(_WIN64)
    return 0;
#else
    return NULL;
#endif
}

// ============================================================================
// Python Type Methods
// ============================================================================

static PyObject *AdaptiveThreadPool_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    AdaptiveThreadPool *self = (AdaptiveThreadPool *)type->tp_alloc(type, 0);
    if (self) {
        self->workers = NULL;
        self->task_queue = NULL;
        self->memory_pool = NULL;
        atomic_init(&self->current_thread_count, 0);
        atomic_init(&self->monitor_running, false);
        atomic_init(&self->shutdown, false);
        thread_lock_init(&self->pool_lock);
    }
    return (PyObject *)self;
}

static int AdaptiveThreadPool_init(AdaptiveThreadPool *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "min_threads", "max_threads", "soft_limit", "hard_limit",
        "policy", "monitor_interval_ms", "scale_up_threshold",
        "scale_down_threshold", "cpu_threshold", NULL
    };
    
    int policy_int = POLICY_BALANCED;
    
    self->min_threads = DEFAULT_MIN_THREADS;
    self->max_threads = DEFAULT_MAX_THREADS;
    self->soft_limit = DEFAULT_SOFT_LIMIT;
    self->hard_limit = DEFAULT_MAX_THREADS;
    self->monitor_interval_ms = DEFAULT_MONITOR_INTERVAL_MS;
    self->scale_up_threshold = DEFAULT_SCALE_UP_THRESHOLD;
    self->scale_down_threshold = DEFAULT_SCALE_DOWN_THRESHOLD;
    self->cpu_threshold = DEFAULT_CPU_THRESHOLD;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|nnnninddd", kwlist,
                                      &self->min_threads, &self->max_threads,
                                      &self->soft_limit, &self->hard_limit,
                                      &policy_int, &self->monitor_interval_ms,
                                      &self->scale_up_threshold,
                                      &self->scale_down_threshold,
                                      &self->cpu_threshold)) {
        return -1;
    }
    
    self->policy = (ScalingPolicy)policy_int;
    
    // Validate configuration
    if (self->min_threads < 1 || self->min_threads > self->max_threads) {
        PyErr_SetString(PyExc_ValueError, "Invalid min_threads value");
        return -1;
    }
    
    if (self->soft_limit > self->hard_limit) {
        PyErr_SetString(PyExc_ValueError, "soft_limit cannot exceed hard_limit");
        return -1;
    }
    
    // Create memory pool
    self->memory_pool = memory_pool_create(MEMORY_POOL_SIZE);
    if (!self->memory_pool) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create memory pool");
        return -1;
    }
    
    // Initialize task queue
    self->task_queue = task_queue_create(TASK_QUEUE_INITIAL_CAPACITY);
    if (!self->task_queue) {
        memory_pool_destroy(self->memory_pool);
        PyErr_SetString(PyExc_MemoryError, "Failed to create task queue");
        return -1;
    }
    
    // Initialize metrics
    atomic_init(&self->metrics.cpu_utilization_percent, 0);
    atomic_init(&self->metrics.memory_available_mb, 0);
    atomic_init(&self->metrics.active_threads, 0);
    atomic_init(&self->metrics.idle_threads, 0);
    atomic_init(&self->metrics.queue_depth, 0);
    atomic_init(&self->metrics.avg_task_duration_ns, 0);
    atomic_init(&self->metrics.tasks_per_second, 0);
    
    // Allocate worker array
    self->workers = calloc(self->hard_limit, sizeof(WorkerThread *));
    if (!self->workers) {
        task_queue_destroy(self->task_queue);
        memory_pool_destroy(self->memory_pool);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate worker array");
        return -1;
    }
    
    // Start initial worker threads
    atomic_store(&self->current_thread_count, 0);
    scale_thread_pool(self, (int)self->min_threads);
    
    // Start monitor thread
    atomic_store(&self->monitor_running, true);
    if (mt_thread_create(&self->monitor_thread, monitor_thread_func, self) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to start monitor thread");
        return -1;
    }
    
    return 0;
}

static void AdaptiveThreadPool_dealloc(AdaptiveThreadPool *self) {
    // Signal shutdown
    atomic_store(&self->shutdown, true);
    atomic_store(&self->monitor_running, false);
    
    // Wait for monitor thread
    if (self->monitor_thread) {
        mt_thread_join(self->monitor_thread);
    }
    
    // Signal all workers to exit
    for (size_t i = 0; i < self->hard_limit; i++) {
        if (self->workers[i]) {
            atomic_store(&self->workers[i]->should_exit, true);
        }
    }
    
    // Destroy task queue (will wake up workers)
    if (self->task_queue) {
        task_queue_destroy(self->task_queue);
    }
    
    // Wait for all workers
    for (size_t i = 0; i < self->hard_limit; i++) {
        if (self->workers[i]) {
            mt_thread_join(self->workers[i]->handle);
            free(self->workers[i]);
        }
    }
    
    free(self->workers);
    thread_lock_destroy(&self->pool_lock);
    
    if (self->memory_pool) {
        memory_pool_destroy(self->memory_pool);
    }
    
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *AdaptiveThreadPool_submit(AdaptiveThreadPool *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"callable", "args", "kwargs", "priority", NULL};
    
    PyObject *callable = NULL;
    PyObject *task_args = NULL;
    PyObject *task_kwargs = NULL;
    unsigned long long priority = UINT64_MAX;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOK", kwlist,
                                      &callable, &task_args, &task_kwargs, &priority)) {
        return NULL;
    }
    
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be callable");
        return NULL;
    }
    
    if (atomic_load(&self->shutdown)) {
        PyErr_SetString(PyExc_RuntimeError, "Thread pool is shut down");
        return NULL;
    }
    
    bool success = task_queue_enqueue(self->task_queue, callable, task_args, task_kwargs, priority);
    
    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to enqueue task");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject *AdaptiveThreadPool_shutdown(AdaptiveThreadPool *self, PyObject *args) {
    int wait = 1;
    
    if (!PyArg_ParseTuple(args, "|p", &wait)) {
        return NULL;
    }
    
    atomic_store(&self->shutdown, true);
    
    if (wait) {
        // Wait for task queue to drain
        while (atomic_load(&self->task_queue->size) > 0) {
            thread_sleep_ms(10);
        }
    }
    
    Py_RETURN_NONE;
}

static PyObject *AdaptiveThreadPool_get_metrics(AdaptiveThreadPool *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    
    PyDict_SetItemString(dict, "cpu_utilization_percent",
                        PyLong_FromUnsignedLongLong(atomic_load(&self->metrics.cpu_utilization_percent)));
    PyDict_SetItemString(dict, "memory_available_mb",
                        PyLong_FromUnsignedLongLong(atomic_load(&self->metrics.memory_available_mb)));
    PyDict_SetItemString(dict, "active_threads",
                        PyLong_FromUnsignedLongLong(atomic_load(&self->metrics.active_threads)));
    PyDict_SetItemString(dict, "idle_threads",
                        PyLong_FromUnsignedLongLong(atomic_load(&self->metrics.idle_threads)));
    PyDict_SetItemString(dict, "current_threads",
                        PyLong_FromUnsignedLongLong(atomic_load(&self->current_thread_count)));
    PyDict_SetItemString(dict, "queue_depth",
                        PyLong_FromUnsignedLongLong(atomic_load(&self->metrics.queue_depth)));
    PyDict_SetItemString(dict, "avg_task_duration_ms",
                        PyFloat_FromDouble(atomic_load(&self->metrics.avg_task_duration_ns) / 1000000.0));
    PyDict_SetItemString(dict, "total_submitted",
                        PyLong_FromUnsignedLongLong(atomic_load(&self->task_queue->total_submitted)));
    PyDict_SetItemString(dict, "total_completed",
                        PyLong_FromUnsignedLongLong(atomic_load(&self->task_queue->total_completed)));
    
    return dict;
}

static PyObject *AdaptiveThreadPool_get_config(AdaptiveThreadPool *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    
    PyDict_SetItemString(dict, "min_threads", PyLong_FromSize_t(self->min_threads));
    PyDict_SetItemString(dict, "max_threads", PyLong_FromSize_t(self->max_threads));
    PyDict_SetItemString(dict, "soft_limit", PyLong_FromSize_t(self->soft_limit));
    PyDict_SetItemString(dict, "hard_limit", PyLong_FromSize_t(self->hard_limit));
    PyDict_SetItemString(dict, "policy", PyLong_FromLong(self->policy));
    PyDict_SetItemString(dict, "monitor_interval_ms", PyLong_FromUnsignedLong(self->monitor_interval_ms));
    PyDict_SetItemString(dict, "scale_up_threshold", PyFloat_FromDouble(self->scale_up_threshold));
    PyDict_SetItemString(dict, "scale_down_threshold", PyFloat_FromDouble(self->scale_down_threshold));
    PyDict_SetItemString(dict, "cpu_threshold", PyFloat_FromDouble(self->cpu_threshold));
    
    return dict;
}

// ============================================================================
// Python Type Definition
// ============================================================================

static PyMethodDef AdaptiveThreadPool_methods[] = {
    {"submit", (PyCFunction)AdaptiveThreadPool_submit, METH_VARARGS | METH_KEYWORDS,
     "Submit a task to the thread pool"},
    {"shutdown", (PyCFunction)AdaptiveThreadPool_shutdown, METH_VARARGS,
     "Shutdown the thread pool"},
    {"get_metrics", (PyCFunction)AdaptiveThreadPool_get_metrics, METH_NOARGS,
     "Get current resource metrics"},
    {"get_config", (PyCFunction)AdaptiveThreadPool_get_config, METH_NOARGS,
     "Get thread pool configuration"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject AdaptiveThreadPoolType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "multithread.AdaptiveThreadPool",
    .tp_doc = "Adaptive thread pool with dynamic scaling",
    .tp_basicsize = sizeof(AdaptiveThreadPool),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = AdaptiveThreadPool_new,
    .tp_init = (initproc)AdaptiveThreadPool_init,
    .tp_dealloc = (destructor)AdaptiveThreadPool_dealloc,
    .tp_methods = AdaptiveThreadPool_methods,
};

// ============================================================================
// Module Definition
// ============================================================================

static PyModuleDef adaptive_threadpool_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "multithread._multithread",
    .m_doc = "Multithread: cross-platform adaptive thread pool for Python 3.14 free-threading (internal)",
    .m_size = -1,
};

/* Init symbol must match .so stem (_multithread) so loader finds it: PyInit__multithread */
PyMODINIT_FUNC PyInit__multithread(void) {
    PyObject *module;
    
    if (PyType_Ready(&AdaptiveThreadPoolType) < 0) {
        return NULL;
    }
    
    module = PyModule_Create(&adaptive_threadpool_module);
    if (!module) {
        return NULL;
    }
    
    Py_INCREF(&AdaptiveThreadPoolType);
    if (PyModule_AddObject(module, "AdaptiveThreadPool", (PyObject *)&AdaptiveThreadPoolType) < 0) {
        Py_DECREF(&AdaptiveThreadPoolType);
        Py_DECREF(module);
        return NULL;
    }
    
    // Add constants
    PyModule_AddIntConstant(module, "POLICY_CONSERVATIVE", POLICY_CONSERVATIVE);
    PyModule_AddIntConstant(module, "POLICY_BALANCED", POLICY_BALANCED);
    PyModule_AddIntConstant(module, "POLICY_AGGRESSIVE", POLICY_AGGRESSIVE);
    PyModule_AddStringConstant(module, "__version__", MODULE_VERSION);
    
    return module;
}
