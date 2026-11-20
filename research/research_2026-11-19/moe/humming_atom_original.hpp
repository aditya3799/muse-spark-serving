// The official CUTLASS BF16 RS warp reordering atom with four MMA atoms across N.
// Its static shape/stride is enumerated against the pinned vendor implementation in atom_probe.py.
#pragma once
#include "cute/tensor.hpp"
using HummingWarpAtom = cute::Layout<
    cute::Shape<cute::Shape<cute::Shape<cute::_8,cute::_2>,cute::_4>,
                cute::Shape<cute::Shape<cute::_2,cute::_4,cute::_2>,cute::_1>>,
    cute::Stride<cute::Stride<cute::Stride<cute::_128,cute::_1>,cute::_8>,
                 cute::Stride<cute::Stride<cute::_4,cute::_32,cute::_2>,cute::_0>>>;
using HummingWeightLayout = decltype(cute::tile_to_shape(
    HummingWarpAtom{},cute::make_shape(int{},int{},cute::_1{})));
