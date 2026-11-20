// Explicit hierarchy for the same frozen Humming WGMMA nibble addresses.
// N: (8,2,4,N/64) @ (128,1,8,1024); K: (2,4,2,K/16) @ (4,32,2,16*N).
#pragma once
#include "cute/tensor.hpp"
using HummingWeightLayout = cute::Layout<
 cute::Shape<cute::Shape<cute::_8,cute::_2,cute::_4,int>,cute::Shape<cute::_2,cute::_4,cute::_2,int>,cute::_1>,
 cute::Stride<cute::Stride<cute::_128,cute::_1,cute::_8,cute::_1024>,cute::Stride<cute::_4,cute::_32,cute::_2,int>,cute::_0>>;
CUTE_HOST_DEVICE HummingWeightLayout make_humming_layout(int n,int k) {
 return cute::make_layout(cute::make_shape(cute::make_shape(cute::_8{},cute::_2{},cute::_4{},n/64),cute::make_shape(cute::_2{},cute::_4{},cute::_2{},k/16),cute::_1{}),
 cute::make_stride(cute::make_stride(cute::_128{},cute::_1{},cute::_8{},cute::_1024{}),cute::make_stride(cute::_4{},cute::_32{},cute::_2{},16*n),cute::_0{}));
}

