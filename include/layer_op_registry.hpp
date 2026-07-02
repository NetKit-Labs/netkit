#pragma once

#include "ops_resolver.hpp"

/*
 * Layer op descriptors for compile-time resolver tables.
 *
 * Each struct exposes constexpr kRegistration (metadata + function pointers).
 * Instantiate NkOpList<YourOps...> to link only the operators your firmware needs.
 */

bool NkPlanDense(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareDense(const NkCnnOpContext& ctx);
void NkEvalDense(CnnBlock& block, const Tensor& input, Tensor& output);

bool NkPlanConv2D(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareConv2D(const NkCnnOpContext& ctx);
void NkEvalConv2D(CnnBlock& block, const Tensor& input, Tensor& output);

bool NkPlanMaxPool2D(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareMaxPool2D(const NkCnnOpContext& ctx);
void NkEvalMaxPool2D(CnnBlock& block, const Tensor& input, Tensor& output);

bool NkPlanAvgPool2D(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareAvgPool2D(const NkCnnOpContext& ctx);
void NkEvalAvgPool2D(CnnBlock& block, const Tensor& input, Tensor& output);

bool NkPlanBatchNorm2d(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareBatchNorm2d(const NkCnnOpContext& ctx);
void NkEvalBatchNorm2d(CnnBlock& block, const Tensor& input, Tensor& output);

bool NkPlanFlatten(CnnBlock& block, NkCnnSpatialPlan& plan);
bool NkPrepareFlatten(const NkCnnOpContext& ctx);
void NkEvalFlatten(CnnBlock& block, const Tensor& input, Tensor& output);

struct NkDenseOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::Dense),
        "dense",
        NkPlanDense,
        NkPrepareDense,
        NkEvalDense,
    };
};

struct NkConv2DOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::Conv2D),
        "conv2d",
        NkPlanConv2D,
        NkPrepareConv2D,
        NkEvalConv2D,
    };
};

struct NkMaxPool2DOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::MaxPool2D),
        "max_pool2d",
        NkPlanMaxPool2D,
        NkPrepareMaxPool2D,
        NkEvalMaxPool2D,
    };
};

struct NkAvgPool2DOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::AvgPool2D),
        "avg_pool2d",
        NkPlanAvgPool2D,
        NkPrepareAvgPool2D,
        NkEvalAvgPool2D,
    };
};

struct NkBatchNorm2dOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::BatchNorm2d),
        "batch_norm2d",
        NkPlanBatchNorm2d,
        NkPrepareBatchNorm2d,
        NkEvalBatchNorm2d,
    };
};

struct NkFlattenOpDescriptor
{
    static constexpr NkLayerOpRegistration kRegistration = {
        static_cast<uint8_t>(NkOpCode::Flatten),
        "flatten",
        NkPlanFlatten,
        NkPrepareFlatten,
        NkEvalFlatten,
    };
};

using NkAllLayerOps = NkOpList<NkDenseOpDescriptor,
                                 NkConv2DOpDescriptor,
                                 NkMaxPool2DOpDescriptor,
                                 NkFlattenOpDescriptor,
                                 NkAvgPool2DOpDescriptor,
                                 NkBatchNorm2dOpDescriptor>;
