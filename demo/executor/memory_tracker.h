#ifndef DEMO_MEMORY_TRACKER_H
#define DEMO_MEMORY_TRACKER_H

#include "allocator.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

struct MemoryStats {
    size_t current_bytes = 0;
    size_t peak_bytes = 0;
    size_t total_allocated_bytes = 0;
    size_t total_freed_bytes = 0;
    size_t allocation_count = 0;
    size_t free_count = 0;
};

class TrackingAllocator : public ncnn::Allocator {
public:
    TrackingAllocator(ncnn::Allocator* backend = nullptr);
    ~TrackingAllocator() override;

    void* fastMalloc(size_t size) override;
    void fastFree(void* ptr) override;

    void clear();
    MemoryStats get_stats() const;
    void reset_peak();

private:
    ncnn::Allocator* backend_;
    mutable std::mutex mutex_;
    std::unordered_map<void*, size_t> allocations_;
    std::atomic<size_t> current_bytes_{0};
    std::atomic<size_t> peak_bytes_{0};
    std::atomic<size_t> total_allocated_bytes_{0};
    std::atomic<size_t> total_freed_bytes_{0};
    std::atomic<size_t> allocation_count_{0};
    std::atomic<size_t> free_count_{0};
};

#endif // DEMO_MEMORY_TRACKER_H
