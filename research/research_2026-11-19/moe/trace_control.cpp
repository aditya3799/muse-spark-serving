
#include <vector>
#include <tuple>
#include <cstdint>
#include <cstdio>
#define MAX(a,b) ((a)>(b)?(a):(b))
#define PRAGMA_UNROLL
#define debug_kernel_timeout_check(x)
#define __syncthreads()
using Event=std::tuple<int,int,int>;
struct Recorder{std::vector<Event> events;void add(int f,int a,int b=0){events.emplace_back(f,a,b);}};
struct Producer{Recorder&r;template<bool A=false,bool B=false>void load_stage(int s,bool yes=true){r.add(0,s,yes);}void load_channel(){r.add(1,0);}};
struct Consumer{Recorder&r;template<bool A=false>void wait_stage(int s){r.add(2,s);} };
struct Pipe{Recorder&r;template<bool A=false>void load_stage_iter(int s,int w){r.add(3,s,w);} };
struct Math{Recorder&r;void run(int s,int w){r.add(4,s,w);}void transform_b(int b,int w){r.add(5,b,w);} };
template<int W>struct Context{static constexpr uint32_t kWarpIters=W;static constexpr bool kUseStreamK=false;};
template<int V>struct Shape{static constexpr int K=V;};
template<int I,int S,int W> std::vector<Event> parent(){
using Ctx=Context<W>; using ProblemShape=Shape<I>;using BlockShape=Shape<1>;constexpr int kNumStages=S;uint32_t slice_iters=I;Recorder rec;Producer producer{rec};Consumer consumer{rec};Pipe s2r_pipe{rec};Math mma{rec};
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0, 0);

    while (slice_iters) {
      debug_kernel_timeout_check(debug_start_clock);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
        debug_kernel_timeout_check(debug_start_clock);
        if (slice_iters == 1) producer.load_channel();
        PRAGMA_UNROLL
        for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
          mma.run(stage_id, warp_iter_id);
          if (warp_iter_id == Ctx::kWarpIters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            } else {
              __syncthreads();
              producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }

          mma.transform_b(
              (warp_iter_id + 1) % 2,
              (warp_iter_id + 1) % Ctx::kWarpIters);
        }

        if constexpr (kNumStages == 2) producer.load_stage(stage_id, slice_iters > kNumStages);
        slice_iters--;
        if (!slice_iters) break;
      };
    };

return rec.events;
}

template<int I,int S,int W> std::vector<Event> cycles(){
using Ctx=Context<W>; using ProblemShape=Shape<I>;using BlockShape=Shape<1>;constexpr int kNumStages=S;uint32_t slice_iters=I;Recorder rec;Producer producer{rec};Consumer consumer{rec};Pipe s2r_pipe{rec};Math mma{rec};
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0, 0);

    static_assert(Ctx::kWarpIters % 2 == 0, "requires alternating operand register banks");
    static_assert(!Ctx::kUseStreamK);
    constexpr uint32_t kSliceIters = ProblemShape::K / BlockShape::K;
    auto consume_stage = [&](uint32_t stage_id, uint32_t remaining_iters) {
        debug_kernel_timeout_check(debug_start_clock);
        if (remaining_iters == 1) producer.load_channel();
        PRAGMA_UNROLL
        for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
          mma.run(stage_id, warp_iter_id);
          if (warp_iter_id == Ctx::kWarpIters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (remaining_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            } else {
              __syncthreads();
              producer.load_stage(stage_id + kNumStages - 1, remaining_iters >= kNumStages);
              if (remaining_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }

          mma.transform_b(
              (warp_iter_id + 1) % 2,
              (warp_iter_id + 1) % Ctx::kWarpIters);
        }

        if constexpr (kNumStages == 2) producer.load_stage(stage_id, remaining_iters > kNumStages);
    };
    #pragma unroll 1
    for (uint32_t cycle = 0; cycle < kSliceIters / kNumStages; ++cycle) {
      PRAGMA_UNROLL
      for (uint32_t stage = 0; stage < kNumStages; ++stage)
        consume_stage(stage, kSliceIters - cycle*kNumStages - stage);
    }
    PRAGMA_UNROLL
    for (uint32_t stage = 0; stage < kSliceIters % kNumStages; ++stage)
      consume_stage(stage, kSliceIters % kNumStages - stage);

return rec.events;
}

template<int I,int S,int W> std::vector<Event> tail(){
using Ctx=Context<W>; using ProblemShape=Shape<I>;using BlockShape=Shape<1>;constexpr int kNumStages=S;uint32_t slice_iters=I;Recorder rec;Producer producer{rec};Consumer consumer{rec};Pipe s2r_pipe{rec};Math mma{rec};
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0, 0);

    static_assert(Ctx::kWarpIters % 2 == 0, "requires alternating operand register banks");
    while (slice_iters) {
      debug_kernel_timeout_check(debug_start_clock);
      PRAGMA_UNROLL
      for (uint32_t stage_id = 0; stage_id < kNumStages; stage_id++) {
        debug_kernel_timeout_check(debug_start_clock);
        if (slice_iters == 1) producer.load_channel();
        PRAGMA_UNROLL
        for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
          if (slice_iters > 1 || warp_iter_id + 1 < Ctx::kWarpIters)
          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
          mma.run(stage_id, warp_iter_id);
          if (warp_iter_id == Ctx::kWarpIters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            } else {
              __syncthreads();
              producer.load_stage(stage_id + kNumStages - 1, slice_iters >= kNumStages);
              if (slice_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }

          if (slice_iters > 1 || warp_iter_id + 1 < Ctx::kWarpIters)
          mma.transform_b(
              (warp_iter_id + 1) % 2,
              (warp_iter_id + 1) % Ctx::kWarpIters);
        }

        if constexpr (kNumStages == 2) producer.load_stage(stage_id, slice_iters > kNumStages);
        slice_iters--;
        if (!slice_iters) break;
      };
    };

return rec.events;
}

template<int I,int S,int W> std::vector<Event> cycles_tail(){
using Ctx=Context<W>; using ProblemShape=Shape<I>;using BlockShape=Shape<1>;constexpr int kNumStages=S;uint32_t slice_iters=I;Recorder rec;Producer producer{rec};Consumer consumer{rec};Pipe s2r_pipe{rec};Math mma{rec};
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0, 0);

    static_assert(Ctx::kWarpIters % 2 == 0, "requires alternating operand register banks");
    static_assert(!Ctx::kUseStreamK);
    constexpr uint32_t kSliceIters = ProblemShape::K / BlockShape::K;
    auto consume_stage = [&](uint32_t stage_id, uint32_t remaining_iters) {
        debug_kernel_timeout_check(debug_start_clock);
        if (remaining_iters == 1) producer.load_channel();
        PRAGMA_UNROLL
        for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
          if (remaining_iters > 1 || warp_iter_id + 1 < Ctx::kWarpIters)
          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
          mma.run(stage_id, warp_iter_id);
          if (warp_iter_id == Ctx::kWarpIters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (remaining_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            } else {
              __syncthreads();
              producer.load_stage(stage_id + kNumStages - 1, remaining_iters >= kNumStages);
              if (remaining_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }

          if (remaining_iters > 1 || warp_iter_id + 1 < Ctx::kWarpIters)
          mma.transform_b(
              (warp_iter_id + 1) % 2,
              (warp_iter_id + 1) % Ctx::kWarpIters);
        }

        if constexpr (kNumStages == 2) producer.load_stage(stage_id, remaining_iters > kNumStages);
    };
    #pragma unroll 1
    for (uint32_t cycle = 0; cycle < kSliceIters / kNumStages; ++cycle) {
      PRAGMA_UNROLL
      for (uint32_t stage = 0; stage < kNumStages; ++stage)
        consume_stage(stage, kSliceIters - cycle*kNumStages - stage);
    }
    PRAGMA_UNROLL
    for (uint32_t stage = 0; stage < kSliceIters % kNumStages; ++stage)
      consume_stage(stage, kSliceIters % kNumStages - stage);

return rec.events;
}

template<int I,int S,int W> std::vector<Event> countdown(){
using Ctx=Context<W>; using ProblemShape=Shape<I>;using BlockShape=Shape<1>;constexpr int kNumStages=S;uint32_t slice_iters=I;Recorder rec;Producer producer{rec};Consumer consumer{rec};Pipe s2r_pipe{rec};Math mma{rec};
    producer.template load_stage<true, true>(0);
    PRAGMA_UNROLL
    for (uint32_t stage_id = 1; stage_id < MAX(kNumStages - 1, 2); stage_id++) {
      producer.load_stage(stage_id, stage_id < slice_iters);
    };

    consumer.template wait_stage<true>(kNumStages);
    s2r_pipe.template load_stage_iter<true>(0, 0);
    mma.transform_b(0, 0);

    static_assert(Ctx::kWarpIters % 2 == 0, "requires alternating operand register banks");
    static_assert(!Ctx::kUseStreamK);
    constexpr uint32_t kSliceIters = ProblemShape::K / BlockShape::K;
    auto consume_stage = [&](uint32_t stage_id, uint32_t remaining_iters) {
        debug_kernel_timeout_check(debug_start_clock);
        if (remaining_iters == 1) producer.load_channel();
        PRAGMA_UNROLL
        for (uint32_t warp_iter_id = 0; warp_iter_id < Ctx::kWarpIters; warp_iter_id++) {
          s2r_pipe.load_stage_iter(stage_id, warp_iter_id + 1);
          mma.run(stage_id, warp_iter_id);
          if (warp_iter_id == Ctx::kWarpIters - 2) {
            if constexpr (kNumStages == 2) {
              __syncthreads();
              if (remaining_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            } else {
              __syncthreads();
              producer.load_stage(stage_id + kNumStages - 1, remaining_iters >= kNumStages);
              if (remaining_iters > 1) consumer.wait_stage((stage_id + 1) % kNumStages);
            }
          }

          mma.transform_b(
              (warp_iter_id + 1) % 2,
              (warp_iter_id + 1) % Ctx::kWarpIters);
        }

        if constexpr (kNumStages == 2) producer.load_stage(stage_id, remaining_iters > kNumStages);
    };
    #pragma unroll 1
    for (uint32_t remaining = kSliceIters; remaining >= kNumStages; remaining -= kNumStages) {
      PRAGMA_UNROLL
      for (uint32_t stage = 0; stage < kNumStages; ++stage)
        consume_stage(stage, remaining - stage);
    }
    PRAGMA_UNROLL
    for (uint32_t stage = 0; stage < kSliceIters % kNumStages; ++stage)
      consume_stage(stage, kSliceIters % kNumStages - stage);

return rec.events;
}

template<int I,int S,int W> int check(){
 auto p=parent<I,S,W>();auto a=cycles<I,S,W>();auto b=tail<I,S,W>();auto c=cycles_tail<I,S,W>();
 if(p!=a||b!=c||p!=countdown<I,S,W>())return 1;
 int last_mma=-1;for(int i=0;i<(int)p.size();++i)if(std::get<0>(p[i])==4)last_mma=i;
 // One lookahead load immediately precedes the final MMA; one unused transform follows.
 std::vector<Event> expected=p;
 if(last_mma<1||std::get<0>(p[last_mma-1])!=3)return 2;
 int last_transform=-1;for(int i=last_mma+1;i<(int)p.size();++i)if(std::get<0>(p[i])==5){last_transform=i;break;}
 if(last_transform<0)return 3;
 expected.erase(expected.begin()+last_transform);expected.erase(expected.begin()+last_mma-1);
 if(expected!=b)return 4;
 int mm=0;for(auto&e:b)mm+=std::get<0>(e)==4;if(mm!=I*W)return 5;return 0;
}
int main(){int cases=0,errors=0;
++cases;{int rc=check<1,2,2>();if(rc){++errors;std::printf("FAIL 1 2 2 %d\n",rc);}}
++cases;{int rc=check<1,2,4>();if(rc){++errors;std::printf("FAIL 1 2 4 %d\n",rc);}}
++cases;{int rc=check<1,2,8>();if(rc){++errors;std::printf("FAIL 1 2 8 %d\n",rc);}}
++cases;{int rc=check<1,3,2>();if(rc){++errors;std::printf("FAIL 1 3 2 %d\n",rc);}}
++cases;{int rc=check<1,3,4>();if(rc){++errors;std::printf("FAIL 1 3 4 %d\n",rc);}}
++cases;{int rc=check<1,3,8>();if(rc){++errors;std::printf("FAIL 1 3 8 %d\n",rc);}}
++cases;{int rc=check<1,4,2>();if(rc){++errors;std::printf("FAIL 1 4 2 %d\n",rc);}}
++cases;{int rc=check<1,4,4>();if(rc){++errors;std::printf("FAIL 1 4 4 %d\n",rc);}}
++cases;{int rc=check<1,4,8>();if(rc){++errors;std::printf("FAIL 1 4 8 %d\n",rc);}}
++cases;{int rc=check<1,6,2>();if(rc){++errors;std::printf("FAIL 1 6 2 %d\n",rc);}}
++cases;{int rc=check<1,6,4>();if(rc){++errors;std::printf("FAIL 1 6 4 %d\n",rc);}}
++cases;{int rc=check<1,6,8>();if(rc){++errors;std::printf("FAIL 1 6 8 %d\n",rc);}}
++cases;{int rc=check<2,2,2>();if(rc){++errors;std::printf("FAIL 2 2 2 %d\n",rc);}}
++cases;{int rc=check<2,2,4>();if(rc){++errors;std::printf("FAIL 2 2 4 %d\n",rc);}}
++cases;{int rc=check<2,2,8>();if(rc){++errors;std::printf("FAIL 2 2 8 %d\n",rc);}}
++cases;{int rc=check<2,3,2>();if(rc){++errors;std::printf("FAIL 2 3 2 %d\n",rc);}}
++cases;{int rc=check<2,3,4>();if(rc){++errors;std::printf("FAIL 2 3 4 %d\n",rc);}}
++cases;{int rc=check<2,3,8>();if(rc){++errors;std::printf("FAIL 2 3 8 %d\n",rc);}}
++cases;{int rc=check<2,4,2>();if(rc){++errors;std::printf("FAIL 2 4 2 %d\n",rc);}}
++cases;{int rc=check<2,4,4>();if(rc){++errors;std::printf("FAIL 2 4 4 %d\n",rc);}}
++cases;{int rc=check<2,4,8>();if(rc){++errors;std::printf("FAIL 2 4 8 %d\n",rc);}}
++cases;{int rc=check<2,6,2>();if(rc){++errors;std::printf("FAIL 2 6 2 %d\n",rc);}}
++cases;{int rc=check<2,6,4>();if(rc){++errors;std::printf("FAIL 2 6 4 %d\n",rc);}}
++cases;{int rc=check<2,6,8>();if(rc){++errors;std::printf("FAIL 2 6 8 %d\n",rc);}}
++cases;{int rc=check<3,2,2>();if(rc){++errors;std::printf("FAIL 3 2 2 %d\n",rc);}}
++cases;{int rc=check<3,2,4>();if(rc){++errors;std::printf("FAIL 3 2 4 %d\n",rc);}}
++cases;{int rc=check<3,2,8>();if(rc){++errors;std::printf("FAIL 3 2 8 %d\n",rc);}}
++cases;{int rc=check<3,3,2>();if(rc){++errors;std::printf("FAIL 3 3 2 %d\n",rc);}}
++cases;{int rc=check<3,3,4>();if(rc){++errors;std::printf("FAIL 3 3 4 %d\n",rc);}}
++cases;{int rc=check<3,3,8>();if(rc){++errors;std::printf("FAIL 3 3 8 %d\n",rc);}}
++cases;{int rc=check<3,4,2>();if(rc){++errors;std::printf("FAIL 3 4 2 %d\n",rc);}}
++cases;{int rc=check<3,4,4>();if(rc){++errors;std::printf("FAIL 3 4 4 %d\n",rc);}}
++cases;{int rc=check<3,4,8>();if(rc){++errors;std::printf("FAIL 3 4 8 %d\n",rc);}}
++cases;{int rc=check<3,6,2>();if(rc){++errors;std::printf("FAIL 3 6 2 %d\n",rc);}}
++cases;{int rc=check<3,6,4>();if(rc){++errors;std::printf("FAIL 3 6 4 %d\n",rc);}}
++cases;{int rc=check<3,6,8>();if(rc){++errors;std::printf("FAIL 3 6 8 %d\n",rc);}}
++cases;{int rc=check<4,2,2>();if(rc){++errors;std::printf("FAIL 4 2 2 %d\n",rc);}}
++cases;{int rc=check<4,2,4>();if(rc){++errors;std::printf("FAIL 4 2 4 %d\n",rc);}}
++cases;{int rc=check<4,2,8>();if(rc){++errors;std::printf("FAIL 4 2 8 %d\n",rc);}}
++cases;{int rc=check<4,3,2>();if(rc){++errors;std::printf("FAIL 4 3 2 %d\n",rc);}}
++cases;{int rc=check<4,3,4>();if(rc){++errors;std::printf("FAIL 4 3 4 %d\n",rc);}}
++cases;{int rc=check<4,3,8>();if(rc){++errors;std::printf("FAIL 4 3 8 %d\n",rc);}}
++cases;{int rc=check<4,4,2>();if(rc){++errors;std::printf("FAIL 4 4 2 %d\n",rc);}}
++cases;{int rc=check<4,4,4>();if(rc){++errors;std::printf("FAIL 4 4 4 %d\n",rc);}}
++cases;{int rc=check<4,4,8>();if(rc){++errors;std::printf("FAIL 4 4 8 %d\n",rc);}}
++cases;{int rc=check<4,6,2>();if(rc){++errors;std::printf("FAIL 4 6 2 %d\n",rc);}}
++cases;{int rc=check<4,6,4>();if(rc){++errors;std::printf("FAIL 4 6 4 %d\n",rc);}}
++cases;{int rc=check<4,6,8>();if(rc){++errors;std::printf("FAIL 4 6 8 %d\n",rc);}}
++cases;{int rc=check<5,2,2>();if(rc){++errors;std::printf("FAIL 5 2 2 %d\n",rc);}}
++cases;{int rc=check<5,2,4>();if(rc){++errors;std::printf("FAIL 5 2 4 %d\n",rc);}}
++cases;{int rc=check<5,2,8>();if(rc){++errors;std::printf("FAIL 5 2 8 %d\n",rc);}}
++cases;{int rc=check<5,3,2>();if(rc){++errors;std::printf("FAIL 5 3 2 %d\n",rc);}}
++cases;{int rc=check<5,3,4>();if(rc){++errors;std::printf("FAIL 5 3 4 %d\n",rc);}}
++cases;{int rc=check<5,3,8>();if(rc){++errors;std::printf("FAIL 5 3 8 %d\n",rc);}}
++cases;{int rc=check<5,4,2>();if(rc){++errors;std::printf("FAIL 5 4 2 %d\n",rc);}}
++cases;{int rc=check<5,4,4>();if(rc){++errors;std::printf("FAIL 5 4 4 %d\n",rc);}}
++cases;{int rc=check<5,4,8>();if(rc){++errors;std::printf("FAIL 5 4 8 %d\n",rc);}}
++cases;{int rc=check<5,6,2>();if(rc){++errors;std::printf("FAIL 5 6 2 %d\n",rc);}}
++cases;{int rc=check<5,6,4>();if(rc){++errors;std::printf("FAIL 5 6 4 %d\n",rc);}}
++cases;{int rc=check<5,6,8>();if(rc){++errors;std::printf("FAIL 5 6 8 %d\n",rc);}}
++cases;{int rc=check<7,2,2>();if(rc){++errors;std::printf("FAIL 7 2 2 %d\n",rc);}}
++cases;{int rc=check<7,2,4>();if(rc){++errors;std::printf("FAIL 7 2 4 %d\n",rc);}}
++cases;{int rc=check<7,2,8>();if(rc){++errors;std::printf("FAIL 7 2 8 %d\n",rc);}}
++cases;{int rc=check<7,3,2>();if(rc){++errors;std::printf("FAIL 7 3 2 %d\n",rc);}}
++cases;{int rc=check<7,3,4>();if(rc){++errors;std::printf("FAIL 7 3 4 %d\n",rc);}}
++cases;{int rc=check<7,3,8>();if(rc){++errors;std::printf("FAIL 7 3 8 %d\n",rc);}}
++cases;{int rc=check<7,4,2>();if(rc){++errors;std::printf("FAIL 7 4 2 %d\n",rc);}}
++cases;{int rc=check<7,4,4>();if(rc){++errors;std::printf("FAIL 7 4 4 %d\n",rc);}}
++cases;{int rc=check<7,4,8>();if(rc){++errors;std::printf("FAIL 7 4 8 %d\n",rc);}}
++cases;{int rc=check<7,6,2>();if(rc){++errors;std::printf("FAIL 7 6 2 %d\n",rc);}}
++cases;{int rc=check<7,6,4>();if(rc){++errors;std::printf("FAIL 7 6 4 %d\n",rc);}}
++cases;{int rc=check<7,6,8>();if(rc){++errors;std::printf("FAIL 7 6 8 %d\n",rc);}}
++cases;{int rc=check<8,2,2>();if(rc){++errors;std::printf("FAIL 8 2 2 %d\n",rc);}}
++cases;{int rc=check<8,2,4>();if(rc){++errors;std::printf("FAIL 8 2 4 %d\n",rc);}}
++cases;{int rc=check<8,2,8>();if(rc){++errors;std::printf("FAIL 8 2 8 %d\n",rc);}}
++cases;{int rc=check<8,3,2>();if(rc){++errors;std::printf("FAIL 8 3 2 %d\n",rc);}}
++cases;{int rc=check<8,3,4>();if(rc){++errors;std::printf("FAIL 8 3 4 %d\n",rc);}}
++cases;{int rc=check<8,3,8>();if(rc){++errors;std::printf("FAIL 8 3 8 %d\n",rc);}}
++cases;{int rc=check<8,4,2>();if(rc){++errors;std::printf("FAIL 8 4 2 %d\n",rc);}}
++cases;{int rc=check<8,4,4>();if(rc){++errors;std::printf("FAIL 8 4 4 %d\n",rc);}}
++cases;{int rc=check<8,4,8>();if(rc){++errors;std::printf("FAIL 8 4 8 %d\n",rc);}}
++cases;{int rc=check<8,6,2>();if(rc){++errors;std::printf("FAIL 8 6 2 %d\n",rc);}}
++cases;{int rc=check<8,6,4>();if(rc){++errors;std::printf("FAIL 8 6 4 %d\n",rc);}}
++cases;{int rc=check<8,6,8>();if(rc){++errors;std::printf("FAIL 8 6 8 %d\n",rc);}}
++cases;{int rc=check<15,2,2>();if(rc){++errors;std::printf("FAIL 15 2 2 %d\n",rc);}}
++cases;{int rc=check<15,2,4>();if(rc){++errors;std::printf("FAIL 15 2 4 %d\n",rc);}}
++cases;{int rc=check<15,2,8>();if(rc){++errors;std::printf("FAIL 15 2 8 %d\n",rc);}}
++cases;{int rc=check<15,3,2>();if(rc){++errors;std::printf("FAIL 15 3 2 %d\n",rc);}}
++cases;{int rc=check<15,3,4>();if(rc){++errors;std::printf("FAIL 15 3 4 %d\n",rc);}}
++cases;{int rc=check<15,3,8>();if(rc){++errors;std::printf("FAIL 15 3 8 %d\n",rc);}}
++cases;{int rc=check<15,4,2>();if(rc){++errors;std::printf("FAIL 15 4 2 %d\n",rc);}}
++cases;{int rc=check<15,4,4>();if(rc){++errors;std::printf("FAIL 15 4 4 %d\n",rc);}}
++cases;{int rc=check<15,4,8>();if(rc){++errors;std::printf("FAIL 15 4 8 %d\n",rc);}}
++cases;{int rc=check<15,6,2>();if(rc){++errors;std::printf("FAIL 15 6 2 %d\n",rc);}}
++cases;{int rc=check<15,6,4>();if(rc){++errors;std::printf("FAIL 15 6 4 %d\n",rc);}}
++cases;{int rc=check<15,6,8>();if(rc){++errors;std::printf("FAIL 15 6 8 %d\n",rc);}}
++cases;{int rc=check<16,2,2>();if(rc){++errors;std::printf("FAIL 16 2 2 %d\n",rc);}}
++cases;{int rc=check<16,2,4>();if(rc){++errors;std::printf("FAIL 16 2 4 %d\n",rc);}}
++cases;{int rc=check<16,2,8>();if(rc){++errors;std::printf("FAIL 16 2 8 %d\n",rc);}}
++cases;{int rc=check<16,3,2>();if(rc){++errors;std::printf("FAIL 16 3 2 %d\n",rc);}}
++cases;{int rc=check<16,3,4>();if(rc){++errors;std::printf("FAIL 16 3 4 %d\n",rc);}}
++cases;{int rc=check<16,3,8>();if(rc){++errors;std::printf("FAIL 16 3 8 %d\n",rc);}}
++cases;{int rc=check<16,4,2>();if(rc){++errors;std::printf("FAIL 16 4 2 %d\n",rc);}}
++cases;{int rc=check<16,4,4>();if(rc){++errors;std::printf("FAIL 16 4 4 %d\n",rc);}}
++cases;{int rc=check<16,4,8>();if(rc){++errors;std::printf("FAIL 16 4 8 %d\n",rc);}}
++cases;{int rc=check<16,6,2>();if(rc){++errors;std::printf("FAIL 16 6 2 %d\n",rc);}}
++cases;{int rc=check<16,6,4>();if(rc){++errors;std::printf("FAIL 16 6 4 %d\n",rc);}}
++cases;{int rc=check<16,6,8>();if(rc){++errors;std::printf("FAIL 16 6 8 %d\n",rc);}}
++cases;{int rc=check<17,2,2>();if(rc){++errors;std::printf("FAIL 17 2 2 %d\n",rc);}}
++cases;{int rc=check<17,2,4>();if(rc){++errors;std::printf("FAIL 17 2 4 %d\n",rc);}}
++cases;{int rc=check<17,2,8>();if(rc){++errors;std::printf("FAIL 17 2 8 %d\n",rc);}}
++cases;{int rc=check<17,3,2>();if(rc){++errors;std::printf("FAIL 17 3 2 %d\n",rc);}}
++cases;{int rc=check<17,3,4>();if(rc){++errors;std::printf("FAIL 17 3 4 %d\n",rc);}}
++cases;{int rc=check<17,3,8>();if(rc){++errors;std::printf("FAIL 17 3 8 %d\n",rc);}}
++cases;{int rc=check<17,4,2>();if(rc){++errors;std::printf("FAIL 17 4 2 %d\n",rc);}}
++cases;{int rc=check<17,4,4>();if(rc){++errors;std::printf("FAIL 17 4 4 %d\n",rc);}}
++cases;{int rc=check<17,4,8>();if(rc){++errors;std::printf("FAIL 17 4 8 %d\n",rc);}}
++cases;{int rc=check<17,6,2>();if(rc){++errors;std::printf("FAIL 17 6 2 %d\n",rc);}}
++cases;{int rc=check<17,6,4>();if(rc){++errors;std::printf("FAIL 17 6 4 %d\n",rc);}}
++cases;{int rc=check<17,6,8>();if(rc){++errors;std::printf("FAIL 17 6 8 %d\n",rc);}}
++cases;{int rc=check<31,2,2>();if(rc){++errors;std::printf("FAIL 31 2 2 %d\n",rc);}}
++cases;{int rc=check<31,2,4>();if(rc){++errors;std::printf("FAIL 31 2 4 %d\n",rc);}}
++cases;{int rc=check<31,2,8>();if(rc){++errors;std::printf("FAIL 31 2 8 %d\n",rc);}}
++cases;{int rc=check<31,3,2>();if(rc){++errors;std::printf("FAIL 31 3 2 %d\n",rc);}}
++cases;{int rc=check<31,3,4>();if(rc){++errors;std::printf("FAIL 31 3 4 %d\n",rc);}}
++cases;{int rc=check<31,3,8>();if(rc){++errors;std::printf("FAIL 31 3 8 %d\n",rc);}}
++cases;{int rc=check<31,4,2>();if(rc){++errors;std::printf("FAIL 31 4 2 %d\n",rc);}}
++cases;{int rc=check<31,4,4>();if(rc){++errors;std::printf("FAIL 31 4 4 %d\n",rc);}}
++cases;{int rc=check<31,4,8>();if(rc){++errors;std::printf("FAIL 31 4 8 %d\n",rc);}}
++cases;{int rc=check<31,6,2>();if(rc){++errors;std::printf("FAIL 31 6 2 %d\n",rc);}}
++cases;{int rc=check<31,6,4>();if(rc){++errors;std::printf("FAIL 31 6 4 %d\n",rc);}}
++cases;{int rc=check<31,6,8>();if(rc){++errors;std::printf("FAIL 31 6 8 %d\n",rc);}}
++cases;{int rc=check<32,2,2>();if(rc){++errors;std::printf("FAIL 32 2 2 %d\n",rc);}}
++cases;{int rc=check<32,2,4>();if(rc){++errors;std::printf("FAIL 32 2 4 %d\n",rc);}}
++cases;{int rc=check<32,2,8>();if(rc){++errors;std::printf("FAIL 32 2 8 %d\n",rc);}}
++cases;{int rc=check<32,3,2>();if(rc){++errors;std::printf("FAIL 32 3 2 %d\n",rc);}}
++cases;{int rc=check<32,3,4>();if(rc){++errors;std::printf("FAIL 32 3 4 %d\n",rc);}}
++cases;{int rc=check<32,3,8>();if(rc){++errors;std::printf("FAIL 32 3 8 %d\n",rc);}}
++cases;{int rc=check<32,4,2>();if(rc){++errors;std::printf("FAIL 32 4 2 %d\n",rc);}}
++cases;{int rc=check<32,4,4>();if(rc){++errors;std::printf("FAIL 32 4 4 %d\n",rc);}}
++cases;{int rc=check<32,4,8>();if(rc){++errors;std::printf("FAIL 32 4 8 %d\n",rc);}}
++cases;{int rc=check<32,6,2>();if(rc){++errors;std::printf("FAIL 32 6 2 %d\n",rc);}}
++cases;{int rc=check<32,6,4>();if(rc){++errors;std::printf("FAIL 32 6 4 %d\n",rc);}}
++cases;{int rc=check<32,6,8>();if(rc){++errors;std::printf("FAIL 32 6 8 %d\n",rc);}}
++cases;{int rc=check<33,2,2>();if(rc){++errors;std::printf("FAIL 33 2 2 %d\n",rc);}}
++cases;{int rc=check<33,2,4>();if(rc){++errors;std::printf("FAIL 33 2 4 %d\n",rc);}}
++cases;{int rc=check<33,2,8>();if(rc){++errors;std::printf("FAIL 33 2 8 %d\n",rc);}}
++cases;{int rc=check<33,3,2>();if(rc){++errors;std::printf("FAIL 33 3 2 %d\n",rc);}}
++cases;{int rc=check<33,3,4>();if(rc){++errors;std::printf("FAIL 33 3 4 %d\n",rc);}}
++cases;{int rc=check<33,3,8>();if(rc){++errors;std::printf("FAIL 33 3 8 %d\n",rc);}}
++cases;{int rc=check<33,4,2>();if(rc){++errors;std::printf("FAIL 33 4 2 %d\n",rc);}}
++cases;{int rc=check<33,4,4>();if(rc){++errors;std::printf("FAIL 33 4 4 %d\n",rc);}}
++cases;{int rc=check<33,4,8>();if(rc){++errors;std::printf("FAIL 33 4 8 %d\n",rc);}}
++cases;{int rc=check<33,6,2>();if(rc){++errors;std::printf("FAIL 33 6 2 %d\n",rc);}}
++cases;{int rc=check<33,6,4>();if(rc){++errors;std::printf("FAIL 33 6 4 %d\n",rc);}}
++cases;{int rc=check<33,6,8>();if(rc){++errors;std::printf("FAIL 33 6 8 %d\n",rc);}}
++cases;{int rc=check<63,2,2>();if(rc){++errors;std::printf("FAIL 63 2 2 %d\n",rc);}}
++cases;{int rc=check<63,2,4>();if(rc){++errors;std::printf("FAIL 63 2 4 %d\n",rc);}}
++cases;{int rc=check<63,2,8>();if(rc){++errors;std::printf("FAIL 63 2 8 %d\n",rc);}}
++cases;{int rc=check<63,3,2>();if(rc){++errors;std::printf("FAIL 63 3 2 %d\n",rc);}}
++cases;{int rc=check<63,3,4>();if(rc){++errors;std::printf("FAIL 63 3 4 %d\n",rc);}}
++cases;{int rc=check<63,3,8>();if(rc){++errors;std::printf("FAIL 63 3 8 %d\n",rc);}}
++cases;{int rc=check<63,4,2>();if(rc){++errors;std::printf("FAIL 63 4 2 %d\n",rc);}}
++cases;{int rc=check<63,4,4>();if(rc){++errors;std::printf("FAIL 63 4 4 %d\n",rc);}}
++cases;{int rc=check<63,4,8>();if(rc){++errors;std::printf("FAIL 63 4 8 %d\n",rc);}}
++cases;{int rc=check<63,6,2>();if(rc){++errors;std::printf("FAIL 63 6 2 %d\n",rc);}}
++cases;{int rc=check<63,6,4>();if(rc){++errors;std::printf("FAIL 63 6 4 %d\n",rc);}}
++cases;{int rc=check<63,6,8>();if(rc){++errors;std::printf("FAIL 63 6 8 %d\n",rc);}}
++cases;{int rc=check<64,2,2>();if(rc){++errors;std::printf("FAIL 64 2 2 %d\n",rc);}}
++cases;{int rc=check<64,2,4>();if(rc){++errors;std::printf("FAIL 64 2 4 %d\n",rc);}}
++cases;{int rc=check<64,2,8>();if(rc){++errors;std::printf("FAIL 64 2 8 %d\n",rc);}}
++cases;{int rc=check<64,3,2>();if(rc){++errors;std::printf("FAIL 64 3 2 %d\n",rc);}}
++cases;{int rc=check<64,3,4>();if(rc){++errors;std::printf("FAIL 64 3 4 %d\n",rc);}}
++cases;{int rc=check<64,3,8>();if(rc){++errors;std::printf("FAIL 64 3 8 %d\n",rc);}}
++cases;{int rc=check<64,4,2>();if(rc){++errors;std::printf("FAIL 64 4 2 %d\n",rc);}}
++cases;{int rc=check<64,4,4>();if(rc){++errors;std::printf("FAIL 64 4 4 %d\n",rc);}}
++cases;{int rc=check<64,4,8>();if(rc){++errors;std::printf("FAIL 64 4 8 %d\n",rc);}}
++cases;{int rc=check<64,6,2>();if(rc){++errors;std::printf("FAIL 64 6 2 %d\n",rc);}}
++cases;{int rc=check<64,6,4>();if(rc){++errors;std::printf("FAIL 64 6 4 %d\n",rc);}}
++cases;{int rc=check<64,6,8>();if(rc){++errors;std::printf("FAIL 64 6 8 %d\n",rc);}}
++cases;{int rc=check<65,2,2>();if(rc){++errors;std::printf("FAIL 65 2 2 %d\n",rc);}}
++cases;{int rc=check<65,2,4>();if(rc){++errors;std::printf("FAIL 65 2 4 %d\n",rc);}}
++cases;{int rc=check<65,2,8>();if(rc){++errors;std::printf("FAIL 65 2 8 %d\n",rc);}}
++cases;{int rc=check<65,3,2>();if(rc){++errors;std::printf("FAIL 65 3 2 %d\n",rc);}}
++cases;{int rc=check<65,3,4>();if(rc){++errors;std::printf("FAIL 65 3 4 %d\n",rc);}}
++cases;{int rc=check<65,3,8>();if(rc){++errors;std::printf("FAIL 65 3 8 %d\n",rc);}}
++cases;{int rc=check<65,4,2>();if(rc){++errors;std::printf("FAIL 65 4 2 %d\n",rc);}}
++cases;{int rc=check<65,4,4>();if(rc){++errors;std::printf("FAIL 65 4 4 %d\n",rc);}}
++cases;{int rc=check<65,4,8>();if(rc){++errors;std::printf("FAIL 65 4 8 %d\n",rc);}}
++cases;{int rc=check<65,6,2>();if(rc){++errors;std::printf("FAIL 65 6 2 %d\n",rc);}}
++cases;{int rc=check<65,6,4>();if(rc){++errors;std::printf("FAIL 65 6 4 %d\n",rc);}}
++cases;{int rc=check<65,6,8>();if(rc){++errors;std::printf("FAIL 65 6 8 %d\n",rc);}}
++cases;{int rc=check<96,2,2>();if(rc){++errors;std::printf("FAIL 96 2 2 %d\n",rc);}}
++cases;{int rc=check<96,2,4>();if(rc){++errors;std::printf("FAIL 96 2 4 %d\n",rc);}}
++cases;{int rc=check<96,2,8>();if(rc){++errors;std::printf("FAIL 96 2 8 %d\n",rc);}}
++cases;{int rc=check<96,3,2>();if(rc){++errors;std::printf("FAIL 96 3 2 %d\n",rc);}}
++cases;{int rc=check<96,3,4>();if(rc){++errors;std::printf("FAIL 96 3 4 %d\n",rc);}}
++cases;{int rc=check<96,3,8>();if(rc){++errors;std::printf("FAIL 96 3 8 %d\n",rc);}}
++cases;{int rc=check<96,4,2>();if(rc){++errors;std::printf("FAIL 96 4 2 %d\n",rc);}}
++cases;{int rc=check<96,4,4>();if(rc){++errors;std::printf("FAIL 96 4 4 %d\n",rc);}}
++cases;{int rc=check<96,4,8>();if(rc){++errors;std::printf("FAIL 96 4 8 %d\n",rc);}}
++cases;{int rc=check<96,6,2>();if(rc){++errors;std::printf("FAIL 96 6 2 %d\n",rc);}}
++cases;{int rc=check<96,6,4>();if(rc){++errors;std::printf("FAIL 96 6 4 %d\n",rc);}}
++cases;{int rc=check<96,6,8>();if(rc){++errors;std::printf("FAIL 96 6 8 %d\n",rc);}}
++cases;{int rc=check<127,2,2>();if(rc){++errors;std::printf("FAIL 127 2 2 %d\n",rc);}}
++cases;{int rc=check<127,2,4>();if(rc){++errors;std::printf("FAIL 127 2 4 %d\n",rc);}}
++cases;{int rc=check<127,2,8>();if(rc){++errors;std::printf("FAIL 127 2 8 %d\n",rc);}}
++cases;{int rc=check<127,3,2>();if(rc){++errors;std::printf("FAIL 127 3 2 %d\n",rc);}}
++cases;{int rc=check<127,3,4>();if(rc){++errors;std::printf("FAIL 127 3 4 %d\n",rc);}}
++cases;{int rc=check<127,3,8>();if(rc){++errors;std::printf("FAIL 127 3 8 %d\n",rc);}}
++cases;{int rc=check<127,4,2>();if(rc){++errors;std::printf("FAIL 127 4 2 %d\n",rc);}}
++cases;{int rc=check<127,4,4>();if(rc){++errors;std::printf("FAIL 127 4 4 %d\n",rc);}}
++cases;{int rc=check<127,4,8>();if(rc){++errors;std::printf("FAIL 127 4 8 %d\n",rc);}}
++cases;{int rc=check<127,6,2>();if(rc){++errors;std::printf("FAIL 127 6 2 %d\n",rc);}}
++cases;{int rc=check<127,6,4>();if(rc){++errors;std::printf("FAIL 127 6 4 %d\n",rc);}}
++cases;{int rc=check<127,6,8>();if(rc){++errors;std::printf("FAIL 127 6 8 %d\n",rc);}}
++cases;{int rc=check<128,2,2>();if(rc){++errors;std::printf("FAIL 128 2 2 %d\n",rc);}}
++cases;{int rc=check<128,2,4>();if(rc){++errors;std::printf("FAIL 128 2 4 %d\n",rc);}}
++cases;{int rc=check<128,2,8>();if(rc){++errors;std::printf("FAIL 128 2 8 %d\n",rc);}}
++cases;{int rc=check<128,3,2>();if(rc){++errors;std::printf("FAIL 128 3 2 %d\n",rc);}}
++cases;{int rc=check<128,3,4>();if(rc){++errors;std::printf("FAIL 128 3 4 %d\n",rc);}}
++cases;{int rc=check<128,3,8>();if(rc){++errors;std::printf("FAIL 128 3 8 %d\n",rc);}}
++cases;{int rc=check<128,4,2>();if(rc){++errors;std::printf("FAIL 128 4 2 %d\n",rc);}}
++cases;{int rc=check<128,4,4>();if(rc){++errors;std::printf("FAIL 128 4 4 %d\n",rc);}}
++cases;{int rc=check<128,4,8>();if(rc){++errors;std::printf("FAIL 128 4 8 %d\n",rc);}}
++cases;{int rc=check<128,6,2>();if(rc){++errors;std::printf("FAIL 128 6 2 %d\n",rc);}}
++cases;{int rc=check<128,6,4>();if(rc){++errors;std::printf("FAIL 128 6 4 %d\n",rc);}}
++cases;{int rc=check<128,6,8>();if(rc){++errors;std::printf("FAIL 128 6 8 %d\n",rc);}}
++cases;{int rc=check<129,2,2>();if(rc){++errors;std::printf("FAIL 129 2 2 %d\n",rc);}}
++cases;{int rc=check<129,2,4>();if(rc){++errors;std::printf("FAIL 129 2 4 %d\n",rc);}}
++cases;{int rc=check<129,2,8>();if(rc){++errors;std::printf("FAIL 129 2 8 %d\n",rc);}}
++cases;{int rc=check<129,3,2>();if(rc){++errors;std::printf("FAIL 129 3 2 %d\n",rc);}}
++cases;{int rc=check<129,3,4>();if(rc){++errors;std::printf("FAIL 129 3 4 %d\n",rc);}}
++cases;{int rc=check<129,3,8>();if(rc){++errors;std::printf("FAIL 129 3 8 %d\n",rc);}}
++cases;{int rc=check<129,4,2>();if(rc){++errors;std::printf("FAIL 129 4 2 %d\n",rc);}}
++cases;{int rc=check<129,4,4>();if(rc){++errors;std::printf("FAIL 129 4 4 %d\n",rc);}}
++cases;{int rc=check<129,4,8>();if(rc){++errors;std::printf("FAIL 129 4 8 %d\n",rc);}}
++cases;{int rc=check<129,6,2>();if(rc){++errors;std::printf("FAIL 129 6 2 %d\n",rc);}}
++cases;{int rc=check<129,6,4>();if(rc){++errors;std::printf("FAIL 129 6 4 %d\n",rc);}}
++cases;{int rc=check<129,6,8>();if(rc){++errors;std::printf("FAIL 129 6 8 %d\n",rc);}}
std::printf("CASES %d ERRORS %d\n",cases,errors);return errors?1:0;}
