#include "memory_tracker.h"

#include <cstdlib>

TrackingAllocator::TrackingAllocator(ncnn::Allocator* backend)
    : backend_(backend)
{
}

TrackingAllocator::~TrackingAllocator() = default;

void* TrackingAllocator::fastMalloc(size_t size)
{
    void* ptr = backend_ ? backend_->fastMalloc(size) : ncnn::fastMalloc(size);
    if (ptr)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            allocations_[ptr] = size;
        }

        size_t current = current_bytes_.fetch_add(size, std::memory_order_relaxed) + size;
        total_allocated_bytes_.fetch_add(size, std::memory_order_relaxed);
        size_t prev_peak = peak_bytes_.load(std::memory_order_relaxed);
        while (current > prev_peak &&
               !peak_bytes_.compare_exchange_weak(prev_peak, current,
                                                  std::memory_order_relaxed, std::memory_order_relaxed)) {
        }
        allocation_count_.fetch_add(1, std::memory_order_relaxed);
    }
    return ptr;
}

void TrackingAllocator::fastFree(void* ptr)
{
    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::unordered_map<void*, size_t>::iterator it = allocations_.find(ptr);
        if (it != allocations_.end())
        {
            size = it->second;
            allocations_.erase(it);
        }
    }

    if (backend_)
    {
        backend_->fastFree(ptr);
    }
    else
    {
        ncnn::fastFree(ptr);
    }

    if (size)
    {
        current_bytes_.fetch_sub(size, std::memory_order_relaxed);
        total_freed_bytes_.fetch_add(size, std::memory_order_relaxed);
    }
    free_count_.fetch_add(1, std::memory_order_relaxed);
}

void TrackingAllocator::clear()
{
    current_bytes_.store(0, std::memory_order_relaxed);
    peak_bytes_.store(0, std::memory_order_relaxed);
    total_allocated_bytes_.store(0, std::memory_order_relaxed);
    total_freed_bytes_.store(0, std::memory_order_relaxed);
    allocation_count_.store(0, std::memory_order_relaxed);
    free_count_.store(0, std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(mutex_);
    allocations_.clear();
}

MemoryStats TrackingAllocator::get_stats() const
{
    MemoryStats s;
    s.current_bytes = current_bytes_.load(std::memory_order_relaxed);
    s.peak_bytes = peak_bytes_.load(std::memory_order_relaxed);
    s.total_allocated_bytes = total_allocated_bytes_.load(std::memory_order_relaxed);
    s.total_freed_bytes = total_freed_bytes_.load(std::memory_order_relaxed);
    s.allocation_count = allocation_count_.load(std::memory_order_relaxed);
    s.free_count = free_count_.load(std::memory_order_relaxed);
    return s;
}

void TrackingAllocator::reset_peak()
{
    peak_bytes_.store(current_bytes_.load(std::memory_order_relaxed), std::memory_order_relaxed);
}
