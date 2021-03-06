import '../common/__node-resolve:empty-d9702ff5.js';
import '../common/process-2545f00a.js';
import { c5 as KernelBackend, c6 as DataStorage, c7 as engine, e as env, c8 as isString, c9 as encodeString, ca as decodeString, ay as buffer, cb as now, bC as whereImpl$1, cc as registerBackend, cd as Elu, ce as LeakyRelu, y as sizeFromShape, cf as getTypedArrayFromDType, cg as Prelu, ch as Relu, ci as Relu6, cj as Reshape, ck as inferFromImplicitShape, f as assert, G as BatchMatMul, cl as computeStrides, bS as _FusedMatMul, N as Acos, P as Acosh, Q as AddN, S as All, aR as parseAxisParam, cm as getAxesPermutation, cn as getInnerMostAxes, co as assertAxesAreInnerMostDims, cp as computeOutAndReduceShapes, c3 as makeZerosTypedArray, aS as expandShapeToKeepDim, U as Any, V as ArgMax, W as ArgMin, X as Asin, Y as Asinh, Z as Atan, _ as Atan2, $ as Atanh, a2 as AvgPool, a0 as eitherStridesOrDilationsAreOne, cq as computePool2DInfo, n as arraysEqual, a3 as AvgPool3D, cr as computePool3DInfo, cs as AvgPool3DGrad, ct as AvgPoolGrad, aa as FusedBatchNorm, a9 as BatchToSpaceND, ab as Bincount, ae as ClipByValue, L as ComplexAbs, aE as Imag, a5 as Concat, af as Conv2D, cu as convertConv2DDataFormat, bK as computeConv2DInfo, bi as TensorBuffer, bH as Conv2DBackpropFilter, ag as Conv2DBackpropInput, ah as Conv3D, cv as computeConv3DInfo, cw as Conv3DBackpropFilterV2, cx as Conv3DBackpropInputV2, ai as Cos, aj as Cosh, bT as CropAndResize, ak as Cumsum, cy as upcastType, cz as getUndoAxesPermutation, al as DenseBincount, am as DepthToSpace, an as DepthwiseConv2dNative, bP as DepthwiseConv2dNativeBackpropFilter, bQ as DepthwiseConv2dNativeBackpropInput, cA as Diag, ao as Dilation2D, cB as computeDilation2DInfo, cC as getArrayFromDType, cD as locToIndex, cE as toTypedArray, cF as Dilation2DBackpropFilter, t as toNestedArray, cG as makeZerosNestedTypedArray, cH as Dilation2DBackpropInput, cI as EluGrad, aq as Equal, au as Erf, aw as ExpandDims, R as RealDiv, cJ as createScalarValue, bn as FFT, az as Fill, cK as inferDtype, bU as FlipLeftRight, K as FloorDiv, bL as FusedConv2D, bR as FusedDepthwiseConv2D, bG as GatherNd, aB as GatherV2, aD as GreaterEqual, bo as IFFT, cL as IsFinite, cM as IsInf, cN as IsNan, aG as LessEqual, aH as LinSpace, aK as Log1p, aT as LogicalAnd, aU as LogicalNot, aV as LogicalOr, aI as LRN, cO as LRNGrad, aP as Max, aW as MaxPool, aX as MaxPool3D, cP as MaxPool3DGrad, cQ as MaxPoolGrad, aY as MaxPoolWithArgmax, cR as Sum, a_ as Mean, a$ as Min, b1 as MirrorPad, cS as indexToLoc, b2 as Mod, bm as Softmax, b3 as Multinomial, bW as NonMaxSuppressionV3, bX as nonMaxSuppressionV3Impl$1, b_ as NonMaxSuppressionV4, b$ as nonMaxSuppressionV4Impl$1, bY as NonMaxSuppressionV5, bZ as nonMaxSuppressionV5Impl$1, O as OneHot, at as ZerosLike, b6 as OnesLike, bt as Pack, ar as assertShapesMatch, b7 as PadV2, b9 as Pow, bb as Range, bd as Reciprocal, c0 as ResizeBilinear, cT as ResizeBilinearGrad, c1 as ResizeNearestNeighbor, cU as ResizeNearestNeighborGrad, be as Reverse, bV as RotateWithOffset, bf as Round, bE as ScatterNd, cV as calculateShapes, as as Select, bh as Selu, a6 as Sigmoid, bj as Sign, bk as Sin, bl as Sinh, aO as Softplus, b8 as SpaceToBatchND, bF as SparseToDense, bp as SplitV, bq as Sqrt, cW as Square, cX as Step, bu as StridedSlice, bv as Tan, a8 as Tanh, ac as Tile, bz as TopK, bA as Unique, bB as Unpack, cY as UnsortedSegmentSum, cZ as registerKernel } from '../common/non_max_suppression_impl-886052e2.js';
import { w as warn, m as mergeRealAndImagArrays, g as getReshaped, a as getPermuted, b as getReshapedPermuted, c as getSliceBeginCoords, d as getSliceSize, e as computeOutShape, f as assertParamsConsistent, E as ERF_A1, h as ERF_A2, i as ERF_A3, j as ERF_A4, k as ERF_A5, l as ERF_P, n as getComplexWithIndex, s as splitRealAndImagArrays, o as complexWithEvenIndex, p as complexWithOddIndex, q as exponents, r as assignToTypedArray, t as exponent, u as prepareAndValidate, v as collectGatherOpShapeInfo, x as getImageCenter, S as SELU_SCALEALPHA, y as SELU_SCALE, z as prepareSplitSize, A as sliceInfo } from '../common/backend_util-5208330b.js';
import { s as seedrandom } from '../common/index-50ec6044.js';
import { a as assertNotComplex, u as unaryKernelFunc, c as createSimpleBinaryKernelImpl, i as identity, b as add, t as transpose, d as binaryKernelFunc, s as slice, e as bincountImpl, r as real, f as complex, g as concatImpl, h as bincountReduceImpl, m as multiply, j as sub, k as gatherV2Impl, l as linSpaceImpl, n as transposeImpl, o as maxImpl, p as cast, z as zeros, q as exp, v as rangeImpl, w as stridedSliceImpl, x as tileImpl, y as topKImpl, A as uniqueImpl, B as absConfig, C as addConfig, D as castConfig, E as ceilConfig, F as complexConfig, G as expConfig, H as expm1Config, I as floorConfig, J as greaterConfig, K as identityConfig, L as lessConfig, M as logConfig, N as maximumConfig, O as minimumConfig, P as multiplyConfig, Q as negConfig, R as notEqualConfig, S as prodConfig, T as realConfig, U as rsqrtConfig, V as sliceConfig, W as squaredDifferenceConfig, X as subConfig, Y as transposeConfig } from '../common/shared-38adcd68.js';
export { Z as shared } from '../common/shared-38adcd68.js';

/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const whereImpl = whereImpl$1;
class MathBackendCPU extends KernelBackend {
    constructor() {
        super();
        this.blockSize = 48;
        this.firstUse = true;
        this.data = new DataStorage(this, engine());
    }
    write(values, shape, dtype) {
        if (this.firstUse) {
            this.firstUse = false;
            if (env().get('IS_NODE')) {
                warn('\n============================\n' +
                    'Hi there ????. Looks like you are running TensorFlow.js in ' +
                    'Node.js. To speed things up dramatically, install our node ' +
                    'backend, which binds to TensorFlow C++, by running ' +
                    'npm i @tensorflow/tfjs-node, ' +
                    'or npm i @tensorflow/tfjs-node-gpu if you have CUDA. ' +
                    'Then call require(\'@tensorflow/tfjs-node\'); (-gpu ' +
                    'suffix for CUDA) at the start of your program. ' +
                    'Visit https://github.com/tensorflow/tfjs-node for more details.' +
                    '\n============================');
            }
        }
        const dataId = {};
        this.data.set(dataId, { values, dtype, refCount: 1 });
        return dataId;
    }
    /**
     * Create a data bucket in cpu backend.
     * @param shape Shape of the `TensorInfo`.
     * @param dtype DType of the `TensorInfo`.
     * @param values The value of the `TensorInfo` stored as a flattened array.
     */
    makeTensorInfo(shape, dtype, values) {
        let outId;
        if (dtype === 'string' && values != null && values.length > 0 &&
            isString(values[0])) {
            const encodedValues = values.map(d => encodeString(d));
            outId = this.write(encodedValues, shape, dtype);
        }
        else {
            outId = this.write(values, shape, dtype);
        }
        return { dataId: outId, shape, dtype };
    }
    /** Increase refCount of a `TensorData`. */
    incRef(dataId) {
        const tensorData = this.data.get(dataId);
        tensorData.refCount++;
    }
    /** Decrease refCount of a `TensorData`. */
    decRef(dataId) {
        if (this.data.has(dataId)) {
            const tensorData = this.data.get(dataId);
            tensorData.refCount--;
        }
    }
    move(dataId, values, shape, dtype) {
        this.data.set(dataId, { values, dtype, refCount: 1 });
    }
    numDataIds() {
        return this.data.numDataIds();
    }
    async read(dataId) {
        return this.readSync(dataId);
    }
    readSync(dataId) {
        const { dtype, complexTensorInfos } = this.data.get(dataId);
        if (dtype === 'complex64') {
            const realValues = this.readSync(complexTensorInfos.real.dataId);
            const imagValues = this.readSync(complexTensorInfos.imag.dataId);
            return mergeRealAndImagArrays(realValues, imagValues);
        }
        return this.data.get(dataId).values;
    }
    bufferSync(t) {
        const data = this.readSync(t.dataId);
        let decodedData = data;
        if (t.dtype === 'string') {
            try {
                // Decode the bytes into string.
                decodedData = data.map(d => decodeString(d));
            }
            catch (_a) {
                throw new Error('Failed to decode encoded string bytes into utf-8');
            }
        }
        return buffer(t.shape, t.dtype, decodedData);
    }
    makeOutput(values, shape, dtype) {
        const dataId = this.write(values, shape, dtype);
        return engine().makeTensorFromDataId(dataId, shape, dtype, this);
    }
    disposeData(dataId) {
        if (this.data.has(dataId)) {
            const { complexTensorInfos } = this.data.get(dataId);
            if (complexTensorInfos != null) {
                this.disposeData(complexTensorInfos.real.dataId);
                this.disposeData(complexTensorInfos.imag.dataId);
            }
            this.data.delete(dataId);
        }
    }
    disposeIntermediateTensorInfo(tensorInfo) {
        const dataId = tensorInfo.dataId;
        if (this.data.has(dataId)) {
            const tensorData = this.data.get(dataId);
            tensorData.refCount--;
            if (tensorData.refCount < 1) {
                this.disposeData(dataId);
            }
        }
    }
    async time(f) {
        const start = now();
        f();
        const kernelMs = now() - start;
        return { kernelMs };
    }
    memory() {
        return {
            // Unreliable due to automatic gc. The numbers above are cumulative.
            unreliable: true,
            reasons: ['The reported memory is an upper bound. Due to automatic garbage ' +
                    'collection, the true allocated memory may be less.']
        };
    }
    where(condition) {
        assertNotComplex([condition], 'where');
        const condVals = this.readSync(condition.dataId);
        return whereImpl(condition.shape, condVals);
    }
    dispose() { }
    floatPrecision() {
        return 32;
    }
    /** Returns the smallest representable number.  */
    epsilon() {
        return super.epsilon();
    }
}

/** @license See the LICENSE file. */
// This code is auto-generated, do not modify this file!
const version = '3.0.0';

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Side effects for default initialization of MathBackendCPU
registerBackend('cpu', () => new MathBackendCPU(), 1 /* priority */);

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const elu = unaryKernelFunc(Elu, (xi) => xi >= 0 ? xi : (Math.exp(xi) - 1));
const eluConfig = {
    kernelName: Elu,
    backendName: 'cpu',
    kernelFunc: elu,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function leakyRelu(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { alpha } = attrs;
    assertNotComplex([x], 'leakyRelu');
    const xSize = sizeFromShape(x.shape);
    const xVals = backend.data.get(x.dataId).values;
    const outVals = getTypedArrayFromDType('float32', xSize);
    for (let i = 0; i < xVals.length; i++) {
        outVals[i] = xVals[i] < 0 ? alpha * xVals[i] : xVals[i];
    }
    return backend.makeTensorInfo(x.shape, 'float32', outVals);
}
const leakyReluConfig = {
    kernelName: LeakyRelu,
    backendName: 'cpu',
    kernelFunc: leakyRelu
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const preluImpl = createSimpleBinaryKernelImpl((xValue, aValue) => xValue < 0 ? aValue * xValue : xValue);
function prelu(args) {
    const { inputs, backend } = args;
    const { x, alpha } = inputs;
    assertNotComplex([x, alpha], 'prelu');
    const aVals = backend.data.get(x.dataId).values;
    const bVals = backend.data.get(alpha.dataId).values;
    const [resultData, resultShape] = preluImpl(x.shape, alpha.shape, aVals, bVals, x.dtype);
    return backend.makeTensorInfo(resultShape, x.dtype, resultData);
}
const preluConfig = {
    kernelName: Prelu,
    backendName: 'cpu',
    kernelFunc: prelu,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const relu = unaryKernelFunc(Relu, (xi) => Math.max(0, xi));
const reluConfig = {
    kernelName: Relu,
    backendName: 'cpu',
    kernelFunc: relu,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const relu6 = unaryKernelFunc(Relu6, (xi) => Math.min(Math.max(0, xi), 6));
const relu6Config = {
    kernelName: Relu6,
    backendName: 'cpu',
    kernelFunc: relu6,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function applyActivation(backend, x, activation, preluActivationWeights, leakyreluAlpha) {
    if (activation === 'linear') {
        return identity({ inputs: { x }, backend });
    }
    else if (activation === 'relu') {
        return relu({ inputs: { x }, backend });
    }
    else if (activation === 'elu') {
        return elu({ inputs: { x }, backend });
    }
    else if (activation === 'relu6') {
        return relu6({ inputs: { x }, backend });
    }
    else if (activation === 'prelu') {
        return prelu({ inputs: { x, alpha: preluActivationWeights }, backend });
    }
    else if (activation === 'leakyrelu') {
        return leakyRelu({ inputs: { x }, backend, attrs: { alpha: leakyreluAlpha } });
    }
    throw new Error(`Activation ${activation} has not been implemented for the CPU backend.`);
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function reshape(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { shape } = attrs;
    const xSize = sizeFromShape(x.shape);
    const $shape = inferFromImplicitShape(shape, xSize);
    const $xSize = sizeFromShape($shape);
    assert(xSize === $xSize, () => `The new shape (${$shape}) has ${$xSize} elements and the old ` +
        `shape (${x.shape}) has ${xSize} elements. The new shape and old ` +
        `shape must have the same number of elements.`);
    backend.incRef(x.dataId);
    const xData = backend.data.get(x.dataId);
    if (xData.complexTensorInfos != null) {
        const real = xData.complexTensorInfos.real;
        const imag = xData.complexTensorInfos.imag;
        real.shape = $shape;
        imag.shape = $shape;
    }
    return { dataId: x.dataId, shape: $shape, dtype: x.dtype };
}
const reshapeConfig = {
    kernelName: Reshape,
    backendName: 'cpu',
    kernelFunc: reshape
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function batchMatMul(args) {
    const { inputs, backend, attrs } = args;
    const { a, b } = inputs;
    const { transposeA, transposeB } = attrs;
    assertNotComplex([a, b], 'matMul');
    const aRank = a.shape.length;
    const bRank = b.shape.length;
    const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
    const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];
    const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
    const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];
    const outerDimsA = a.shape.slice(0, -2);
    const outerDimsB = b.shape.slice(0, -2);
    const batchDimA = sizeFromShape(outerDimsA);
    const batchDimB = sizeFromShape(outerDimsB);
    const batchDimsCompatible = batchDimA === batchDimB || batchDimA === 1 || batchDimB === 1;
    assert(aRank >= 2 && bRank >= 2 && batchDimsCompatible, () => `Error in matMul: the input batch dimensions must either be the ` +
        `same or at least one input batch dimension must be 1. Got input ` +
        `batch dimensions of (${outerDimsA}) and (${outerDimsB}).`);
    const outShapeOuterDims = batchDimA > batchDimB ? a.shape.slice(0, -2) : b.shape.slice(0, -2);
    const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);
    assert(innerShapeA === innerShapeB, () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
        `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
        `${b.shape} and transposeA=${transposeA}` +
        ` and transposeB=${transposeB} must match.`);
    const a3dShape = transposeA ? [batchDimA, innerShapeA, outerShapeA] :
        [batchDimA, outerShapeA, innerShapeA];
    const b3dShape = transposeB ? [batchDimB, outerShapeB, innerShapeB] :
        [batchDimB, innerShapeB, outerShapeB];
    // The rest of the implementation is designed to operate on rank-3 tensors
    const a3d = reshape({ inputs: { x: a }, backend, attrs: { shape: a3dShape } });
    const b3d = reshape({ inputs: { x: b }, backend, attrs: { shape: b3dShape } });
    const sharedDim = transposeA ? a3d.shape[1] : a3d.shape[2];
    const leftDim = transposeA ? a3d.shape[2] : a3d.shape[1];
    const rightDim = transposeB ? b3d.shape[1] : b3d.shape[2];
    const batchDim = Math.max(batchDimA, batchDimB);
    const a3dValues = backend.data.get(a3d.dataId).values;
    const b3dValues = backend.data.get(b3d.dataId).values;
    const a3dStrides = computeStrides(a3d.shape);
    const b3dStrides = computeStrides(b3d.shape);
    const [aBatch, aOuterStep, aInnerStep] = transposeA ?
        [a3dStrides[0], 1, a3dStrides[1]] :
        [a3dStrides[0], a3dStrides[1], 1];
    const [bInnerStep, bOuterStep, bBatch] = transposeB ?
        [1, b3dStrides[1], b3dStrides[0]] :
        [b3dStrides[1], 1, b3dStrides[0]];
    const size = leftDim * rightDim;
    const result = buffer([batchDim, leftDim, rightDim], a3d.dtype);
    const resVals = result.values;
    const blockSize = backend.blockSize;
    for (let bi = 0; bi < batchDim; bi++) {
        for (let i0 = 0; i0 < leftDim; i0 += blockSize) {
            for (let j0 = 0; j0 < rightDim; j0 += blockSize) {
                for (let k0 = 0; k0 < sharedDim; k0 += blockSize) {
                    // for when blockSize doesn't evenly divide the input
                    const iBlock = Math.min(i0 + blockSize, leftDim);
                    const jBlock = Math.min(j0 + blockSize, rightDim);
                    const kBlock = Math.min(k0 + blockSize, sharedDim);
                    for (let i = i0; i < iBlock; i++) {
                        for (let j = j0; j < jBlock; j++) {
                            let sum = 0.0;
                            for (let k = k0; k < kBlock; k++) {
                                const batchOffsetA = Math.min(bi, batchDimA - 1) * aBatch;
                                const batchOffsetB = Math.min(bi, batchDimB - 1) * bBatch;
                                const aVal = a3dValues[batchOffsetA + i * aOuterStep + k * aInnerStep];
                                const bVal = b3dValues[k * bInnerStep + j * bOuterStep + batchOffsetB];
                                sum += aVal * bVal;
                            }
                            resVals[bi * size + (i * rightDim + j)] += sum;
                        }
                    }
                }
            }
        }
    }
    backend.disposeIntermediateTensorInfo(a3d);
    backend.disposeIntermediateTensorInfo(b3d);
    // set correct shape on output.
    return backend.makeTensorInfo(outShape, result.dtype, result.values);
}
const batchMatMulConfig = {
    kernelName: BatchMatMul,
    backendName: 'cpu',
    kernelFunc: batchMatMul,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function _fusedMatMul(args) {
    const { inputs, backend, attrs } = args;
    const { a, b, bias, preluActivationWeights } = inputs;
    const { transposeA, transposeB, activation, leakyreluAlpha } = attrs;
    let current;
    let addRes;
    let activationRes;
    const intermediates = [];
    const matMulRes = batchMatMul({ inputs: { a, b }, attrs: { transposeA, transposeB }, backend });
    current = matMulRes;
    if (bias) {
        addRes = add({ inputs: { a: current, b: bias }, backend });
        intermediates.push(current);
        current = addRes;
    }
    if (activation) {
        activationRes = applyActivation(backend, current, activation, preluActivationWeights, leakyreluAlpha);
        intermediates.push(current);
        current = activationRes;
    }
    for (const i of intermediates) {
        backend.disposeIntermediateTensorInfo(i);
    }
    return current;
}
const _fusedMatMulConfig = {
    kernelName: _FusedMatMul,
    backendName: 'cpu',
    kernelFunc: _fusedMatMul,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const acos = unaryKernelFunc(Acos, (xi) => Math.acos(xi));
const acosConfig = {
    kernelName: Acos,
    backendName: 'cpu',
    kernelFunc: acos,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const acosh = unaryKernelFunc(Acosh, (xi) => Math.acosh(xi));
const acoshConfig = {
    kernelName: Acosh,
    backendName: 'cpu',
    kernelFunc: acosh,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function addN(args) {
    const { inputs, backend } = args;
    const tensors = inputs;
    assertNotComplex(inputs, 'addN');
    const vals = tensors.map(t => backend.data.get(t.dataId).values);
    const outBuf = buffer(tensors[0].shape, tensors[0].dtype);
    const outVals = outBuf.values;
    for (let i = 0; i < tensors.length; i++) {
        const currVals = vals[i];
        for (let j = 0; j < outVals.length; j++) {
            outVals[j] += currVals[j];
        }
    }
    return backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
}
const addNConfig = {
    kernelName: AddN,
    backendName: 'cpu',
    kernelFunc: addN
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function all(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    assertNotComplex(x, 'all');
    const origAxes = parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = getAxesPermutation(axes, x.shape.length);
    let $x = x;
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        axes = getInnerMostAxes(axes.length, x.shape.length);
    }
    assertAxesAreInnerMostDims('all', axes, $x.shape.length);
    const [outShape, reduceShape] = computeOutAndReduceShapes($x.shape, axes);
    const reduceSize = sizeFromShape(reduceShape);
    const vals = makeZerosTypedArray(sizeFromShape(outShape), $x.dtype);
    const aVals = backend.data.get($x.dataId).values;
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let all = aVals[offset];
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            all = all && value;
        }
        vals[i] = all;
    }
    if (permutedAxes != null) {
        backend.disposeIntermediateTensorInfo($x);
    }
    const result = backend.makeTensorInfo(outShape, $x.dtype, vals);
    if (keepDims) {
        const expandedShape = expandShapeToKeepDim(outShape, origAxes);
        const reshapedResult = reshape({ inputs: { x: result }, backend, attrs: { shape: expandedShape } });
        backend.disposeIntermediateTensorInfo(result);
        return reshapedResult;
    }
    return result;
}
const allConfig = {
    kernelName: All,
    backendName: 'cpu',
    kernelFunc: all
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function any(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    assertNotComplex(x, 'any');
    const origAxes = parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = getAxesPermutation(axes, x.shape.length);
    let $x = x;
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        axes = getInnerMostAxes(axes.length, x.shape.length);
    }
    assertAxesAreInnerMostDims('any', axes, $x.shape.length);
    const [outShape, reduceShape] = computeOutAndReduceShapes($x.shape, axes);
    const reduceSize = sizeFromShape(reduceShape);
    const vals = makeZerosTypedArray(sizeFromShape(outShape), $x.dtype);
    const aVals = backend.data.get($x.dataId).values;
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let anyVal = aVals[offset];
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            anyVal = anyVal || value;
        }
        vals[i] = anyVal;
    }
    if (permutedAxes != null) {
        backend.disposeIntermediateTensorInfo($x);
    }
    const result = backend.makeTensorInfo(outShape, $x.dtype, vals);
    if (keepDims) {
        const expandedShape = expandShapeToKeepDim(outShape, origAxes);
        const reshapedResult = reshape({ inputs: { x: result }, backend, attrs: { shape: expandedShape } });
        backend.disposeIntermediateTensorInfo(result);
        return reshapedResult;
    }
    return result;
}
const anyConfig = {
    kernelName: Any,
    backendName: 'cpu',
    kernelFunc: any
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function argMax(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis } = attrs;
    assertNotComplex(x, 'argMax');
    let axes = parseAxisParam(axis, x.shape);
    const permutedAxes = getAxesPermutation(axes, x.shape.length);
    let $x = x;
    const intermediateTensorInfos = [];
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        intermediateTensorInfos.push($x);
        axes = getInnerMostAxes(axes.length, $x.shape.length);
    }
    axes = [axes[0]];
    assertAxesAreInnerMostDims('argMax', axes, $x.shape.length);
    const [outShape, reduceShape] = computeOutAndReduceShapes($x.shape, axes);
    const outSize = sizeFromShape(outShape);
    const vals = makeZerosTypedArray(outSize, 'int32');
    const reduceSize = sizeFromShape(reduceShape);
    const aVals = backend.data.get($x.dataId).values;
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let max = aVals[offset];
        let maxIndex = 0;
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            if (value > max) {
                max = value;
                maxIndex = j;
            }
        }
        vals[i] = maxIndex;
    }
    intermediateTensorInfos.forEach(t => backend.disposeIntermediateTensorInfo(t));
    return backend.makeTensorInfo(outShape, 'int32', vals);
}
const argMaxConfig = {
    kernelName: ArgMax,
    backendName: 'cpu',
    kernelFunc: argMax
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function argMin(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis } = attrs;
    assertNotComplex(x, 'argMin');
    let axes = parseAxisParam(axis, x.shape);
    const permutedAxes = getAxesPermutation(axes, x.shape.length);
    let $x = x;
    const intermediateTensorInfos = [];
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        intermediateTensorInfos.push($x);
        axes = getInnerMostAxes(axes.length, $x.shape.length);
    }
    axes = [axes[0]];
    assertAxesAreInnerMostDims('argMin', axes, $x.shape.length);
    const [outShape, reduceShape] = computeOutAndReduceShapes($x.shape, axes);
    const outSize = sizeFromShape(outShape);
    const vals = makeZerosTypedArray(outSize, 'int32');
    const reduceSize = sizeFromShape(reduceShape);
    const aVals = backend.data.get($x.dataId).values;
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let min = aVals[offset];
        let minIndex = 0;
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            if (value < min) {
                min = value;
                minIndex = j;
            }
        }
        vals[i] = minIndex;
    }
    intermediateTensorInfos.forEach(t => backend.disposeIntermediateTensorInfo(t));
    return backend.makeTensorInfo(outShape, 'int32', vals);
}
const argMinConfig = {
    kernelName: ArgMin,
    backendName: 'cpu',
    kernelFunc: argMin
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const asin = unaryKernelFunc(Asin, (xi) => Math.asin(xi));
const asinConfig = {
    kernelName: Asin,
    backendName: 'cpu',
    kernelFunc: asin,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const asinh = unaryKernelFunc(Asinh, (xi) => Math.asinh(xi));
const asinhConfig = {
    kernelName: Asinh,
    backendName: 'cpu',
    kernelFunc: asinh,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const atan = unaryKernelFunc(Atan, (xi) => Math.atan(xi));
const atanConfig = {
    kernelName: Atan,
    backendName: 'cpu',
    kernelFunc: atan,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const atan2Impl = createSimpleBinaryKernelImpl((aValue, bValue) => Math.atan2(aValue, bValue));
const atan2 = binaryKernelFunc(Atan2, atan2Impl);
const atan2Config = {
    kernelName: Atan2,
    backendName: 'cpu',
    kernelFunc: atan2,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const atanh = unaryKernelFunc(Atanh, (xi) => Math.atanh(xi));
const atanhConfig = {
    kernelName: Atanh,
    backendName: 'cpu',
    kernelFunc: atanh,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function pool(xValues, xShape, dtype, strides, convInfo, poolType) {
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
        Number.POSITIVE_INFINITY);
    const output = buffer(convInfo.outShape, dtype);
    const outputVals = output.values;
    const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] * convInfo.outShape[3];
    const outputRowStrides = convInfo.outShape[2] * convInfo.outShape[3];
    const outputColStrides = convInfo.outShape[3];
    for (let b = 0; b < convInfo.batchSize; ++b) {
        const outputBatchOffset = b * outputBatchStrides;
        const inputBatchOffset = b * strides[0];
        for (let d = 0; d < convInfo.inChannels; ++d) {
            for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                const xRCorner = yR * strideHeight - padTop;
                const xRMin = Math.max(0, xRCorner);
                const xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                const outputRowOffset = outputBatchOffset + yR * outputRowStrides;
                for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                    const xCCorner = yC * strideWidth - padLeft;
                    const xCMin = Math.max(0, xCCorner);
                    const xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                    let minMaxValue = initialValue;
                    let avgValue = 0;
                    let count = 0;
                    for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
                        const xROffset = inputBatchOffset + xR * strides[1];
                        for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                            const xCOffset = xROffset + xC * strides[2];
                            const pixel = xValues[xCOffset + d];
                            if ((poolType === 'max' && pixel > minMaxValue)) {
                                minMaxValue = pixel;
                            }
                            else if (poolType === 'avg') {
                                avgValue += pixel;
                                count++;
                            }
                        }
                        if (isNaN(minMaxValue)) {
                            break;
                        }
                    }
                    const outputOffset = outputRowOffset + yC * outputColStrides + d;
                    outputVals[outputOffset] =
                        poolType === 'avg' ? avgValue / count : minMaxValue;
                }
            }
        }
    }
    return output;
}
function maxPoolPositions(xValues, xShape, dtype, convInfo, flattenPositions = false, includeBatchInIndex = false) {
    const maxPositions = buffer(convInfo.outShape, 'int32');
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const xBuf = buffer(xShape, dtype, xValues);
    for (let b = 0; b < convInfo.batchSize; ++b) {
        for (let d = 0; d < convInfo.inChannels; ++d) {
            for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                const xRCorner = yR * strideHeight - padTop;
                let xRMin = xRCorner;
                while (xRMin < 0) {
                    xRMin += dilationHeight;
                }
                // const xRMin = Math.max(0, xRCorner);
                const xRMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRCorner);
                for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                    const xCCorner = yC * strideWidth - padLeft;
                    let xCMin = xCCorner;
                    while (xCMin < 0) {
                        xCMin += dilationWidth;
                    }
                    const xCMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xCCorner);
                    let maxValue = Number.NEGATIVE_INFINITY;
                    let maxPosition = -1;
                    for (let xR = xRMin; xR < xRMax; xR += dilationHeight) {
                        const wR = xR - xRCorner;
                        for (let xC = xCMin; xC < xCMax; xC += dilationWidth) {
                            const wC = xC - xCCorner;
                            const pixel = xBuf.get(b, xR, xC, d);
                            if (pixel > maxValue) {
                                maxValue = pixel;
                                if (flattenPositions) {
                                    maxPosition = includeBatchInIndex ?
                                        ((b * convInfo.inHeight + xR) * convInfo.inWidth + xC) *
                                            convInfo.inChannels +
                                            d :
                                        (xR * convInfo.inWidth + xC) * convInfo.inChannels + d;
                                }
                                else {
                                    maxPosition = wR * effectiveFilterWidth + wC;
                                }
                            }
                        }
                    }
                    maxPositions.set(maxPosition, b, yR, yC, d);
                }
            }
        }
    }
    return maxPositions;
}
function pool3d(xValues, xShape, dtype, strides, convInfo, poolType) {
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const initialValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
        Number.POSITIVE_INFINITY);
    const output = buffer(convInfo.outShape, dtype);
    const outputVals = output.values;
    const outputBatchStrides = convInfo.outShape[1] * convInfo.outShape[2] *
        convInfo.outShape[3] * convInfo.outShape[4];
    const outputDepthStrides = convInfo.outShape[2] * convInfo.outShape[3] * convInfo.outShape[4];
    const outputRowStrides = convInfo.outShape[3] * convInfo.outShape[4];
    const outputColStrides = convInfo.outShape[4];
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        const outputBatchOffset = batch * outputBatchStrides;
        const inputBatchOffset = batch * strides[0];
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                const xDepthCorner = yDepth * strideDepth - padFront;
                let xDepthMin = xDepthCorner;
                while (xDepthMin < 0) {
                    xDepthMin += dilationDepth;
                }
                const xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                const outputDepthOffset = outputBatchOffset + yDepth * outputDepthStrides;
                for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                    const xRowCorner = yRow * strideHeight - padTop;
                    let xRowMin = xRowCorner;
                    while (xRowMin < 0) {
                        xRowMin += dilationHeight;
                    }
                    const xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                    const outputRowOffset = outputDepthOffset + yRow * outputRowStrides;
                    for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                        const xColCorner = yCol * strideWidth - padLeft;
                        let xColMin = xColCorner;
                        while (xColMin < 0) {
                            xColMin += dilationWidth;
                        }
                        const xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                        // Shader code begins
                        const outputColOffset = outputRowOffset + yCol * outputColStrides;
                        let minMaxValue = initialValue;
                        let avgValue = 0;
                        let count = 0;
                        for (let xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                            const xDepthOffset = inputBatchOffset + xDepth * strides[1];
                            for (let xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                const xRowOffset = xDepthOffset + xRow * strides[2];
                                for (let xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                    const xColOffset = xRowOffset + xCol * strides[3];
                                    const pixel = xValues[xColOffset + channel];
                                    if ((poolType === 'max' && pixel > minMaxValue)) {
                                        minMaxValue = pixel;
                                    }
                                    else if (poolType === 'avg') {
                                        avgValue += pixel;
                                        count++;
                                    }
                                    if (isNaN(minMaxValue)) {
                                        break;
                                    }
                                }
                                if (isNaN(minMaxValue)) {
                                    break;
                                }
                            }
                            if (isNaN(minMaxValue)) {
                                break;
                            }
                        }
                        const outputOffset = outputColOffset + channel;
                        outputVals[outputOffset] =
                            poolType === 'avg' ? avgValue / count : minMaxValue;
                    }
                }
            }
        }
    }
    return output;
}
function maxPool3dPositions(xBuf, convInfo) {
    const maxPositions = buffer(convInfo.outShape, 'int32');
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let yDepth = 0; yDepth < convInfo.outDepth; ++yDepth) {
                const xDepthCorner = yDepth * strideDepth - padFront;
                let xDepthMin = xDepthCorner;
                while (xDepthMin < 0) {
                    xDepthMin += dilationDepth;
                }
                const xDepthMax = Math.min(convInfo.inDepth, effectiveFilterDepth + xDepthCorner);
                for (let yRow = 0; yRow < convInfo.outHeight; ++yRow) {
                    const xRowCorner = yRow * strideHeight - padTop;
                    let xRowMin = xRowCorner;
                    while (xRowMin < 0) {
                        xRowMin += dilationHeight;
                    }
                    const xRowMax = Math.min(convInfo.inHeight, effectiveFilterHeight + xRowCorner);
                    for (let yCol = 0; yCol < convInfo.outWidth; ++yCol) {
                        const xColCorner = yCol * strideWidth - padLeft;
                        let xColMin = xColCorner;
                        while (xColMin < 0) {
                            xColMin += dilationWidth;
                        }
                        const xColMax = Math.min(convInfo.inWidth, effectiveFilterWidth + xColCorner);
                        // Shader code begins
                        let maxValue = Number.NEGATIVE_INFINITY;
                        let maxPosition = -1;
                        for (let xDepth = xDepthMin; xDepth < xDepthMax; xDepth += dilationDepth) {
                            const wDepth = xDepth - xDepthCorner;
                            for (let xRow = xRowMin; xRow < xRowMax; xRow += dilationHeight) {
                                const wRow = xRow - xRowCorner;
                                for (let xCol = xColMin; xCol < xColMax; xCol += dilationWidth) {
                                    const wCol = xCol - xColCorner;
                                    const pixel = xBuf.get(batch, xDepth, xRow, xCol, channel);
                                    if (pixel >= maxValue) {
                                        maxValue = pixel;
                                        maxPosition =
                                            wDepth * effectiveFilterHeight * effectiveFilterWidth +
                                                wRow * effectiveFilterHeight + wCol;
                                    }
                                }
                            }
                        }
                        maxPositions.set(maxPosition, batch, yDepth, yRow, yCol, channel);
                    }
                }
            }
        }
    }
    return maxPositions;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function avgPool(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    assertNotComplex(x, 'avgPool');
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const dilations = 1;
    assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in avgPool: Either strides or dilations must be 1. ' +
        `Got strides ${strides} and dilations '${dilations}'`);
    const convInfo = computePool2DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    let res;
    if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
        arraysEqual(convInfo.inShape, convInfo.outShape)) {
        res = identity({ inputs: { x }, backend });
    }
    else {
        const xValues = backend.data.get(x.dataId).values;
        const strides = computeStrides(x.shape);
        const buffer = pool(xValues, x.shape, x.dtype, strides, convInfo, 'avg');
        res = backend.makeTensorInfo(convInfo.outShape, x.dtype, buffer.values);
    }
    return res;
}
const avgPoolConfig = {
    kernelName: AvgPool,
    backendName: 'cpu',
    kernelFunc: avgPool
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function avgPool3D(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dimRoundingMode, dataFormat } = attrs;
    assertNotComplex(x, 'avgPool3d');
    const convInfo = computePool3DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode, dataFormat);
    const xValues = backend.data.get(x.dataId).values;
    const outBuf = pool3d(xValues, x.shape, x.dtype, computeStrides(x.shape), convInfo, 'avg');
    return backend.makeTensorInfo(outBuf.shape, 'float32', outBuf.values);
}
const avgPool3DConfig = {
    kernelName: AvgPool3D,
    backendName: 'cpu',
    kernelFunc: avgPool3D
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function avgPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    assertNotComplex([dy, input], 'avgPool3DGrad');
    const convInfo = computePool3DInfo(input.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterDepth = convInfo.filterDepth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = buffer(input.shape, 'float32');
    const avgMultiplier = 1 / (filterDepth * filterHeight * filterWidth);
    const dyBuf = backend.bufferSync(dy);
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
                for (let dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
                    for (let dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
                        // Shader code begins.
                        const dyDepthCorner = dxDepth - padFront;
                        const dyRowCorner = dxRow - padTop;
                        const dyColCorner = dxCol - padLeft;
                        let dotProd = 0;
                        for (let wDepth = 0; wDepth < effectiveFilterDepth; wDepth += dilationDepth) {
                            const dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                            if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                                Math.floor(dyDepth) !== dyDepth) {
                                continue;
                            }
                            for (let wRow = 0; wRow < effectiveFilterHeight; wRow += dilationHeight) {
                                const dyRow = (dyRowCorner + wRow) / strideHeight;
                                if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                                    Math.floor(dyRow) !== dyRow) {
                                    continue;
                                }
                                for (let wCol = 0; wCol < effectiveFilterWidth; wCol += dilationWidth) {
                                    const dyCol = (dyColCorner + wCol) / strideWidth;
                                    if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                                        Math.floor(dyCol) !== dyCol) {
                                        continue;
                                    }
                                    const pixel = dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                    dotProd += pixel;
                                }
                            }
                        }
                        dx.set(dotProd * avgMultiplier, batch, dxDepth, dxRow, dxCol, channel);
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
const avgPool3DGradConfig = {
    kernelName: AvgPool3DGrad,
    backendName: 'cpu',
    kernelFunc: avgPool3DGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function avgPoolGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const x = input;
    assertNotComplex([dy, input], 'avgPoolGrad');
    const { filterSize, strides, pad } = attrs;
    const convInfo = computePool2DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad);
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = buffer(x.shape, 'float32');
    const avgMultiplier = 1 / (filterHeight * filterWidth);
    const dyData = backend.data.get(dy.dataId).values;
    const dyBuf = buffer(dy.shape, 'float32', dyData);
    for (let b = 0; b < convInfo.batchSize; ++b) {
        for (let d = 0; d < convInfo.inChannels; ++d) {
            for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
                for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
                    // Shader code begins.
                    const dyRCorner = dxR - padTop;
                    const dyCCorner = dxC - padLeft;
                    let dotProd = 0;
                    for (let wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
                        const dyR = (dyRCorner + wR) / strideHeight;
                        if (dyR < 0 || dyR >= convInfo.outHeight ||
                            Math.floor(dyR) !== dyR) {
                            continue;
                        }
                        for (let wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                            const dyC = (dyCCorner + wC) / strideWidth;
                            if (dyC < 0 || dyC >= convInfo.outWidth ||
                                Math.floor(dyC) !== dyC) {
                                continue;
                            }
                            const pixel = dyBuf.get(b, dyR, dyC, d);
                            dotProd += pixel;
                        }
                    }
                    dx.set(dotProd * avgMultiplier, b, dxR, dxC, d);
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
const avgPoolGradConfig = {
    kernelName: AvgPoolGrad,
    backendName: 'cpu',
    kernelFunc: avgPoolGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function batchNorm(args) {
    const { inputs, backend, attrs } = args;
    const { x, scale, offset, mean, variance } = inputs;
    assert(mean.shape.length === variance.shape.length, () => 'Batch normalization gradient requires mean and variance to have ' +
        'equal ranks.');
    assert(offset == null || mean.shape.length === offset.shape.length, () => 'Batch normalization gradient requires mean and offset to have ' +
        'equal ranks.');
    assert(scale == null || mean.shape.length === scale.shape.length, () => 'Batch normalization gradient requires mean and scale to have ' +
        'equal ranks.');
    assertNotComplex([x, mean, variance, scale, offset], 'batchNorm');
    let { varianceEpsilon } = attrs;
    if (varianceEpsilon == null) {
        varianceEpsilon = 0.001;
    }
    const xVals = backend.data.get(x.dataId).values;
    const mVals = backend.data.get(mean.dataId).values;
    const varVals = backend.data.get(variance.dataId).values;
    const sVals = scale ? backend.data.get(scale.dataId).values :
        new Float32Array([1]);
    const offVals = offset ?
        backend.data.get(offset.dataId).values :
        new Float32Array([0]);
    const outVals = new Float32Array(xVals.length);
    const offValsLength = offVals.length;
    const sValsLength = sVals.length;
    const varValsLength = varVals.length;
    const mValsLength = mVals.length;
    let offi = 0;
    let mi = 0;
    let si = 0;
    let vi = 0;
    for (let i = 0; i < xVals.length; ++i) {
        outVals[i] = offVals[offi++] +
            (xVals[i] - mVals[mi++]) * sVals[si++] /
                Math.sqrt(varVals[vi++] + varianceEpsilon);
        if (offi >= offValsLength) {
            offi = 0;
        }
        if (mi >= mValsLength) {
            mi = 0;
        }
        if (si >= sValsLength) {
            si = 0;
        }
        if (vi >= varValsLength) {
            vi = 0;
        }
    }
    return backend.makeTensorInfo(x.shape, x.dtype, outVals);
}
const batchNormConfig = {
    kernelName: FusedBatchNorm,
    backendName: 'cpu',
    kernelFunc: batchNorm,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function batchToSpaceND(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockShape, crops } = attrs;
    assertNotComplex([x], 'batchToSpaceND');
    const prod = blockShape.reduce((a, b) => a * b);
    const reshaped = getReshaped(x.shape, blockShape, prod);
    const permuted = getPermuted(reshaped.length, blockShape.length);
    const reshapedPermuted = getReshapedPermuted(x.shape, blockShape, prod);
    const sliceBeginCoords = getSliceBeginCoords(crops, blockShape.length);
    const sliceSize = getSliceSize(reshapedPermuted, crops, blockShape.length);
    const xReshaped = reshape({ inputs: { x }, backend, attrs: { shape: reshaped } });
    const xTransposed = transpose({ inputs: { x: xReshaped }, backend, attrs: { perm: permuted } });
    const xTransposedReshaped = reshape({ inputs: { x: xTransposed }, backend, attrs: { shape: reshapedPermuted } });
    const result = slice({
        inputs: { x: xTransposedReshaped },
        backend,
        attrs: { begin: sliceBeginCoords, size: sliceSize }
    });
    backend.disposeIntermediateTensorInfo(xReshaped);
    backend.disposeIntermediateTensorInfo(xTransposed);
    backend.disposeIntermediateTensorInfo(xTransposedReshaped);
    return result;
}
const batchToSpaceNDConfig = {
    kernelName: BatchToSpaceND,
    backendName: 'cpu',
    kernelFunc: batchToSpaceND
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function bincount(args) {
    const { inputs, backend, attrs } = args;
    const { x, weights } = inputs;
    const { size } = attrs;
    const xVals = backend.data.get(x.dataId).values;
    const weightsVals = backend.data.get(weights.dataId).values;
    const outVals = bincountImpl(xVals, weightsVals, weights.dtype, weights.shape, size);
    return backend.makeTensorInfo([size], weights.dtype, outVals);
}
const bincountConfig = {
    kernelName: Bincount,
    backendName: 'cpu',
    kernelFunc: bincount
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const clip = unaryKernelFunc(ClipByValue, (xi, attrs) => {
    const clipAttrs = attrs;
    if (xi > clipAttrs.clipValueMax) {
        return clipAttrs.clipValueMax;
    }
    return xi < clipAttrs.clipValueMin ? clipAttrs.clipValueMin : xi;
});
const clipConfig = {
    kernelName: ClipByValue,
    backendName: 'cpu',
    kernelFunc: clip,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const complexAbs = (args) => {
    const { x } = args.inputs;
    const cpuBackend = args.backend;
    const resultValues = new Float32Array(sizeFromShape(x.shape));
    const complexVals = cpuBackend.data.get(x.dataId);
    const real = complexVals.complexTensorInfos.real;
    const imag = complexVals.complexTensorInfos.imag;
    const realVals = cpuBackend.data.get(real.dataId).values;
    const imagVals = cpuBackend.data.get(imag.dataId).values;
    for (let i = 0; i < realVals.length; i++) {
        const real = realVals[i];
        const imag = imagVals[i];
        resultValues[i] = Math.hypot(real, imag);
    }
    return cpuBackend.makeOutput(resultValues, x.shape, 'float32');
};
const complexAbsConfig = {
    kernelName: ComplexAbs,
    backendName: 'cpu',
    kernelFunc: complexAbs,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function imag(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const imag = backend.data.get(input.dataId).complexTensorInfos.imag;
    const imagVal = backend.data.get(imag.dataId).values;
    // When complex tensor is disposed, its underlying parts will be disposed too.
    // Make new tensor out of the imag value of the complex. This makes sure the
    // value is still accessible even if complex tensor is disposed.
    return backend.makeTensorInfo(imag.shape, imag.dtype, imagVal);
}
const imagConfig = {
    kernelName: Imag,
    backendName: 'cpu',
    kernelFunc: imag
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function concat(args) {
    const { inputs, backend, attrs } = args;
    const { axis } = attrs;
    const $axis = parseAxisParam(axis, inputs[0].shape)[0];
    let outShape = computeOutShape(inputs.map(t => t.shape), $axis);
    if (sizeFromShape(outShape) === 0) {
        return backend.makeTensorInfo(outShape, inputs[0].dtype, []);
    }
    // Keep only non-empty tensors (ignore tensors with 0 in their shape).
    const $inputs = inputs.filter(t => sizeFromShape(t.shape) > 0);
    if ($inputs.length === 1) {
        return identity({ inputs: { x: $inputs[0] }, backend });
    }
    const shapes = $inputs.map(t => t.shape);
    assertParamsConsistent(shapes, $axis);
    if ($inputs[0].dtype === 'complex64') {
        const reals = $inputs.map((t) => real({ inputs: { input: t }, backend }));
        const imags = $inputs.map((t) => imag({ inputs: { input: t }, backend }));
        const realConcated = concat({ inputs: reals, backend, attrs: { axis: $axis } });
        const imagConcated = concat({ inputs: imags, backend, attrs: { axis: $axis } });
        const result = complex({ inputs: { real: realConcated, imag: imagConcated }, backend });
        reals.forEach(r => backend.disposeIntermediateTensorInfo(r));
        imags.forEach(i => backend.disposeIntermediateTensorInfo(i));
        backend.disposeIntermediateTensorInfo(realConcated);
        backend.disposeIntermediateTensorInfo(imagConcated);
        return result;
    }
    // Any concat of n-dimensional tensors across any axis can be reduced to
    // a concatenation of two-dimensional tensors across the axis 1 by first
    // partitioning the axes of the original tensors into those less than the
    // axis to be concatenated and the rest. Then reshape the tensors
    // into a two-dimensional tensor by collapsing these two sets of axes and
    // concatenate the resulting matrices across the axis 1, finally reshaping
    // the result to have the proper shape.
    const inputs2D = $inputs.map(t => {
        const innerSize = sizeFromShape(t.shape.slice($axis));
        const shape = [-1, innerSize];
        return reshape({ inputs: { x: t }, backend, attrs: { shape } });
    });
    const inputsValShapes = inputs2D.map(t => {
        return { vals: backend.data.get(t.dataId).values, shape: t.shape };
    });
    // Concats 2d tensors along axis=1.
    outShape =
        computeOutShape(inputs2D.map(t => t.shape), 1 /* axis */);
    const simplyConcat = inputs2D[0].shape[0] === 1;
    const outVals = concatImpl(inputsValShapes, outShape, inputs[0].dtype, simplyConcat);
    const finalOutShape = computeOutShape($inputs.map(t => t.shape), $axis);
    const outInfo = backend.makeTensorInfo(finalOutShape, inputs[0].dtype, outVals);
    inputs2D.forEach(t => backend.disposeIntermediateTensorInfo(t));
    return outInfo;
}
const concatConfig = {
    kernelName: Concat,
    backendName: 'cpu',
    kernelFunc: concat
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function conv2D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter } = inputs;
    const { strides, pad, dataFormat, dilations, dimRoundingMode } = attrs;
    assertNotComplex([x, filter], 'conv2d');
    const $dataFormat = convertConv2DDataFormat(dataFormat);
    const convInfo = computeConv2DInfo(x.shape, filter.shape, strides, dilations, pad, dimRoundingMode, false /* depthwise */, $dataFormat);
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const padLeft = convInfo.padInfo.left;
    const padTop = convInfo.padInfo.top;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const y = new TensorBuffer(convInfo.outShape, x.dtype);
    const xStrides = computeStrides(x.shape);
    const filterStrides = computeStrides(filter.shape);
    const xBatchStride = xStrides[0];
    const xRowStride = isChannelsLast ? xStrides[1] : xStrides[2];
    const xColStride = isChannelsLast ? xStrides[2] : 1;
    const xChannelStride = isChannelsLast ? 1 : xStrides[1];
    const yBatchStride = y.strides[0];
    const yRowStride = isChannelsLast ? y.strides[1] : y.strides[2];
    const yColStride = isChannelsLast ? y.strides[2] : 1;
    const yChannelStride = isChannelsLast ? 1 : y.strides[1];
    const xVals = backend.data.get(x.dataId).values;
    const wVals = backend.data.get(filter.dataId).values;
    const yVals = y.values;
    for (let b = 0; b < convInfo.batchSize; ++b) {
        const xOffset1 = b * xBatchStride;
        const yOffset1 = b * yBatchStride;
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
            const yOffset2 = yOffset1 + yR * yRowStride;
            const xRCorner = yR * convInfo.strideHeight - padTop;
            for (let wR = 0; wR < filterHeight; ++wR) {
                const xR = xRCorner + wR * dilationHeight;
                if (xR < 0 || xR >= convInfo.inHeight) {
                    continue;
                }
                const wOffset1 = wR * filterStrides[0];
                const xOffset2 = xOffset1 + xR * xRowStride;
                for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                    const yOffset3 = yOffset2 + yC * yColStride;
                    const xCCorner = yC * convInfo.strideWidth - padLeft;
                    for (let wC = 0; wC < filterWidth; ++wC) {
                        const xC = xCCorner + wC * dilationWidth;
                        if (xC < 0 || xC >= convInfo.inWidth) {
                            continue;
                        }
                        const wOffset2 = wOffset1 + wC * filterStrides[1];
                        const xOffset3 = xOffset2 + xC * xColStride;
                        let wOffset3 = wOffset2;
                        for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                            const xVal = xVals[xOffset3 + d1 * xChannelStride];
                            for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                                yVals[yOffset3 + d2 * yChannelStride] +=
                                    xVal * wVals[wOffset3 + d2];
                            }
                            wOffset3 += convInfo.outChannels;
                        }
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(y.shape, y.dtype, yVals);
}
const conv2DConfig = {
    kernelName: Conv2D,
    backendName: 'cpu',
    kernelFunc: conv2D
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function conv2DBackpropFilter(args) {
    const { inputs, backend, attrs } = args;
    const { x, dy } = inputs;
    const { strides, pad, dataFormat, dimRoundingMode, filterShape } = attrs;
    assertNotComplex([x, dy], 'conv2dBackpropFilter');
    const $dataFormat = convertConv2DDataFormat(dataFormat);
    const convInfo = computeConv2DInfo(x.shape, filterShape, strides, 1 /* dilations */, pad, dimRoundingMode, false /* depthwise */, $dataFormat);
    const { strideHeight, strideWidth, filterHeight, filterWidth } = convInfo;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const dW = new TensorBuffer(convInfo.filterShape, 'float32');
    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;
    const xVals = backend.data.get(x.dataId).values;
    const dyVals = backend.data.get(dy.dataId).values;
    const xBuf = new TensorBuffer(x.shape, x.dtype, xVals);
    const dyBuf = new TensorBuffer(dy.shape, dy.dtype, dyVals);
    for (let wR = 0; wR < filterHeight; ++wR) {
        const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
        const yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
        for (let wC = 0; wC < filterWidth; ++wC) {
            const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
            const yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
            for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                    let dotProd = 0;
                    for (let b = 0; b < convInfo.batchSize; ++b) {
                        for (let yR = yRMin; yR < yRMax; ++yR) {
                            const xR = wR + yR * strideHeight - topPad;
                            for (let yC = yCMin; yC < yCMax; ++yC) {
                                const xC = wC + yC * strideWidth - leftPad;
                                if (isChannelsLast) {
                                    dotProd += xBuf.get(b, xR, xC, d1) *
                                        dyBuf.get(b, yR, yC, d2);
                                }
                                else {
                                    dotProd += xBuf.get(b, d1, xR, xC) *
                                        dyBuf.get(b, d2, yR, yC);
                                }
                            }
                        }
                    }
                    dW.set(dotProd, wR, wC, d1, d2);
                }
            }
        }
    }
    return backend.makeTensorInfo(dW.shape, dW.dtype, dW.values);
}
const conv2DBackpropFilterConfig = {
    kernelName: Conv2DBackpropFilter,
    backendName: 'cpu',
    kernelFunc: conv2DBackpropFilter
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function conv2DBackpropInput(args) {
    const { inputs, backend, attrs } = args;
    const { dy, filter } = inputs;
    const { inputShape, strides, pad, dataFormat, dimRoundingMode } = attrs;
    assertNotComplex([dy, filter], 'conv2dBackpropInput');
    const filterStrides = computeStrides(filter.shape);
    const dyStrides = computeStrides(dy.shape);
    let $dataFormat = convertConv2DDataFormat(dataFormat);
    const convInfo = computeConv2DInfo(inputShape, filter.shape, strides, 1 /* dilations */, pad, dimRoundingMode, false, $dataFormat);
    const dx = new TensorBuffer(convInfo.inShape, 'float32');
    const dxValues = dx.values;
    const dyValues = backend.data.get(dy.dataId).values;
    const fltValues = backend.data.get(filter.dataId).values;
    const [fltS0, fltS1, fltS2] = filterStrides;
    const { batchSize, filterHeight, filterWidth, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, strideHeight, strideWidth } = convInfo;
    $dataFormat = convInfo.dataFormat;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;
    const isChannelsLast = $dataFormat === 'channelsLast';
    const xBatchStride = dx.strides[0];
    const xRowStride = isChannelsLast ? dx.strides[1] : dx.strides[2];
    const xColStride = isChannelsLast ? dx.strides[2] : 1;
    const xChannelStride = isChannelsLast ? 1 : dx.strides[1];
    const yBatchStride = dyStrides[0];
    const yRowStride = isChannelsLast ? dyStrides[1] : dyStrides[2];
    const yColStride = isChannelsLast ? dyStrides[2] : 1;
    const yChannelStride = isChannelsLast ? 1 : dyStrides[1];
    for (let b = 0; b < batchSize; ++b) {
        for (let d1 = 0; d1 < inChannels; ++d1) {
            for (let xR = 0; xR < inHeight; ++xR) {
                const xRCorner = xR - topPad;
                const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                const yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                for (let xC = 0; xC < inWidth; ++xC) {
                    const xCCorner = xC - leftPad;
                    const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                    const yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                    let dotProd = 0;
                    for (let yR = xRMin; yR < yRMax; ++yR) {
                        const wR = yR * strideHeight - xRCorner;
                        for (let yC = xCMin; yC < yCMax; ++yC) {
                            const wC = yC * strideWidth - xCCorner;
                            const dyOffset = yBatchStride * b + yRowStride * yR + yColStride * yC;
                            const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                            for (let d2 = 0; d2 < outChannels; ++d2) {
                                const pixel = dyValues[dyOffset + yChannelStride * d2];
                                const weight = fltValues[fltOffset + d2];
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    const dxOffset = xBatchStride * b + xRowStride * xR +
                        xColStride * xC + xChannelStride * d1;
                    dxValues[dxOffset] = dotProd;
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
const conv2DBackpropInputConfig = {
    kernelName: Conv2DBackpropInput,
    backendName: 'cpu',
    kernelFunc: conv2DBackpropInput
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function conv3D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter } = inputs;
    const { strides, pad, dilations } = attrs;
    assertNotComplex([x, filter], 'conv3d');
    const convInfo = computeConv3DInfo(x.shape, filter.shape, strides, dilations, pad);
    const { filterDepth, filterHeight, filterWidth, dilationDepth, dilationHeight, dilationWidth, padInfo } = convInfo;
    const padFront = padInfo.front;
    const padLeft = padInfo.left;
    const padTop = padInfo.top;
    const y = new TensorBuffer(convInfo.outShape, x.dtype);
    const xVals = backend.data.get(x.dataId).values;
    const wVals = backend.data.get(filter.dataId).values;
    const yVals = y.values;
    const xStrides = computeStrides(x.shape);
    const filterStrides = computeStrides(filter.shape);
    for (let b = 0; b < convInfo.batchSize; ++b) {
        const xOffset1 = b * xStrides[0];
        const yOffset1 = b * y.strides[0];
        for (let yF = 0; yF < convInfo.outDepth; ++yF) {
            const yOffset2 = yOffset1 + yF * y.strides[1];
            const xFCorner = yF * convInfo.strideDepth - padFront;
            for (let wF = 0; wF < filterDepth; ++wF) {
                const xF = xFCorner + wF * dilationDepth;
                if (xF < 0 || xF >= convInfo.inDepth) {
                    continue;
                }
                const wOffset1 = wF * filterStrides[0];
                const xOffset2 = xOffset1 + xF * xStrides[1];
                for (let yR = 0; yR < convInfo.outHeight; ++yR) {
                    const yOffset3 = yOffset2 + yR * y.strides[2];
                    const xRCorner = yR * convInfo.strideHeight - padTop;
                    for (let wR = 0; wR < filterHeight; ++wR) {
                        const xR = xRCorner + wR * dilationHeight;
                        if (xR < 0 || xR >= convInfo.inHeight) {
                            continue;
                        }
                        const wOffset2 = wOffset1 + wR * filterStrides[1];
                        const xOffset3 = xOffset2 + xR * xStrides[2];
                        for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                            const yOffset4 = yOffset3 + yC * convInfo.outChannels;
                            const xCCorner = yC * convInfo.strideWidth - padLeft;
                            for (let wC = 0; wC < filterWidth; ++wC) {
                                const xC = xCCorner + wC * dilationWidth;
                                if (xC < 0 || xC >= convInfo.inWidth) {
                                    continue;
                                }
                                const wOffset3 = wOffset2 + wC * filterStrides[2];
                                const xOffset4 = xOffset3 + xC * convInfo.inChannels;
                                let wOffset4 = wOffset3;
                                for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                                    const xVal = xVals[xOffset4 + d1];
                                    for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                                        yVals[yOffset4 + d2] += xVal * wVals[wOffset4 + d2];
                                    }
                                    wOffset4 += convInfo.outChannels;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(y.shape, y.dtype, y.values);
}
const conv3DConfig = {
    kernelName: Conv3D,
    backendName: 'cpu',
    kernelFunc: conv3D
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function conv3DBackpropFilterV2(args) {
    const { inputs, backend, attrs } = args;
    const { x, dy } = inputs;
    const { strides, pad, filterShape } = attrs;
    assertNotComplex([x, dy], 'conv3dBackpropFilterV2');
    const xStrides = computeStrides(x.shape);
    const dyStrides = computeStrides(dy.shape);
    const convInfo = computeConv3DInfo(x.shape, filterShape, strides, 1 /* dilations */, pad);
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterDepth = convInfo.filterDepth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const dw = new TensorBuffer(convInfo.filterShape, 'float32');
    const dwValues = dw.values;
    const [dwS0, dwS1, dwS2, dwS3] = dw.strides;
    const dyValues = backend.data.get(dy.dataId).values;
    const [dyS0, dyS1, dyS2, dyS3] = dyStrides;
    const xValues = backend.data.get(x.dataId).values;
    const [xS0, xS1, xS2, xS3] = xStrides;
    const frontPad = convInfo.padInfo.front;
    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;
    for (let wF = 0; wF < filterDepth; ++wF) {
        const yFMin = Math.max(0, Math.ceil((frontPad - wF) / strideDepth));
        const yFMax = Math.min(convInfo.outDepth, (convInfo.inDepth + frontPad - wF) / strideDepth);
        const wOffset1 = wF * dwS0;
        for (let wR = 0; wR < filterHeight; ++wR) {
            const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
            const yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
            const wOffset2 = wR * dwS1 + wOffset1;
            for (let wC = 0; wC < filterWidth; ++wC) {
                const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                const yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
                const wOffset3 = wC * dwS2 + wOffset2;
                for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                    const wOffset4 = d1 * dwS3 + wOffset3;
                    for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                        let dotProd = 0;
                        for (let b = 0; b < convInfo.batchSize; ++b) {
                            const xOffset1 = b * xS0;
                            const yOffset1 = b * dyS0;
                            for (let yF = yFMin; yF < yFMax; ++yF) {
                                const xF = wF + yF * strideDepth - frontPad;
                                const xOffset2 = xF * xS1 + xOffset1;
                                const yOffset2 = yF * dyS1 + yOffset1;
                                for (let yR = yRMin; yR < yRMax; ++yR) {
                                    const xR = wR + yR * strideHeight - topPad;
                                    const xOffset3 = xR * xS2 + xOffset2;
                                    const yOffset3 = yR * dyS2 + yOffset2;
                                    for (let yC = yCMin; yC < yCMax; ++yC) {
                                        const xC = wC + yC * strideWidth - leftPad;
                                        const xOffset4 = xC * xS3 + xOffset3;
                                        const yOffset4 = yC * dyS3 + yOffset3;
                                        dotProd += xValues[xOffset4 + d1] * dyValues[yOffset4 + d2];
                                    }
                                }
                            }
                        }
                        dwValues[wOffset4 + d2] = dotProd;
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(dw.shape, dw.dtype, dw.values);
}
const conv3DBackpropFilterV2Config = {
    kernelName: Conv3DBackpropFilterV2,
    backendName: 'cpu',
    kernelFunc: conv3DBackpropFilterV2
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function conv3DBackpropInputV2(args) {
    const { inputs, backend, attrs } = args;
    const { dy, filter } = inputs;
    const { pad, strides, inputShape } = attrs;
    assertNotComplex([dy], 'conv3dBackpropInputV2');
    const dyStrides = computeStrides(dy.shape);
    const filterStrides = computeStrides(filter.shape);
    const convInfo = computeConv3DInfo(inputShape, filter.shape, strides, 1 /* dilations */, pad);
    const dx = new TensorBuffer(convInfo.inShape, 'float32');
    const dxValues = dx.values;
    const [dxS0, dxS1, dxS2, dxS3] = dx.strides;
    const dyValues = backend.data.get(dy.dataId).values;
    const [dyS0, dyS1, dyS2, dyS3] = dyStrides;
    const fltValues = backend.data.get(filter.dataId).values;
    const [fltS0, fltS1, fltS2, fltS3] = filterStrides;
    const { batchSize, filterDepth, filterHeight, filterWidth, inChannels, inDepth, inHeight, inWidth, outChannels, outDepth, outHeight, outWidth, strideDepth, strideHeight, strideWidth } = convInfo;
    const frontPad = filterDepth - 1 - convInfo.padInfo.front;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;
    for (let b = 0; b < batchSize; ++b) {
        for (let d1 = 0; d1 < inChannels; ++d1) {
            // Frames of depth
            for (let xF = 0; xF < inDepth; ++xF) {
                const xFCorner = xF - frontPad;
                const xFMin = Math.max(0, Math.ceil(xFCorner / strideDepth));
                const yFMax = Math.min(outDepth, (filterDepth + xFCorner) / strideDepth);
                // Rows as per standard 2d matrix notation
                for (let xR = 0; xR < inHeight; ++xR) {
                    const xRCorner = xR - topPad;
                    const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                    const yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                    // Columns as per standard 2d matrix notation
                    for (let xC = 0; xC < inWidth; ++xC) {
                        const xCCorner = xC - leftPad;
                        const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                        const yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                        let dotProd = 0;
                        for (let yF = xFMin; yF < yFMax; ++yF) {
                            const wF = yF * strideDepth - xFCorner;
                            for (let yR = xRMin; yR < yRMax; ++yR) {
                                const wR = yR * strideHeight - xRCorner;
                                for (let yC = xCMin; yC < yCMax; ++yC) {
                                    const wC = yC * strideWidth - xCCorner;
                                    const dyOffset = dyS0 * b + dyS1 * yF + dyS2 * yR + dyS3 * yC;
                                    const fltOffset = fltS0 * (filterDepth - 1 - wF) +
                                        fltS1 * (filterHeight - 1 - wR) +
                                        fltS2 * (filterWidth - 1 - wC) + fltS3 * d1;
                                    for (let d2 = 0; d2 < outChannels; ++d2) {
                                        const pixel = dyValues[dyOffset + d2];
                                        const weight = fltValues[fltOffset + d2];
                                        dotProd += pixel * weight;
                                    }
                                }
                            }
                        }
                        dxValues[dxS0 * b + dxS1 * xF + dxS2 * xR + dxS3 * xC + d1] =
                            dotProd;
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
const conv3DBackpropInputV2Config = {
    kernelName: Conv3DBackpropInputV2,
    backendName: 'cpu',
    kernelFunc: conv3DBackpropInputV2
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const cos = unaryKernelFunc(Cos, (xi) => Math.cos(xi));
const cosConfig = {
    kernelName: Cos,
    backendName: 'cpu',
    kernelFunc: cos,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const cosh = unaryKernelFunc(Cosh, (xi) => Math.cosh(xi));
const coshConfig = {
    kernelName: Cosh,
    backendName: 'cpu',
    kernelFunc: cosh,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function cropAndResize(args) {
    const { inputs, backend, attrs } = args;
    const { image, boxes, boxInd } = inputs;
    const { cropSize, method, extrapolationValue } = attrs;
    const [batch, imageHeight, imageWidth, numChannels] = image.shape;
    const numBoxes = boxes.shape[0];
    const [cropHeight, cropWidth] = cropSize;
    const output = buffer([numBoxes, cropHeight, cropWidth, numChannels], 'float32');
    const boxVals = backend.data.get(boxes.dataId).values;
    const boxIndVals = backend.data.get(boxInd.dataId).values;
    const imageVals = backend.data.get(image.dataId).values;
    const inStride = computeStrides(image.shape); // to calculate flat indexes into image
    const outStride = computeStrides(output.shape); // to calculate flat indexes into output
    // Reference implementation
    // tslint:disable-next-line:max-line-length
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/crop_and_resize_op.cc
    for (let b = 0; b < numBoxes; b++) {
        const startInd = b * 4;
        const y1 = boxVals[startInd];
        const x1 = boxVals[startInd + 1];
        const y2 = boxVals[startInd + 2];
        const x2 = boxVals[startInd + 3];
        const bInd = boxIndVals[b];
        if (bInd >= batch) {
            continue;
        }
        const heightScale = (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : 0;
        const widthScale = (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;
        for (let y = 0; y < cropHeight; y++) {
            const yInd = (cropHeight > 1) ?
                y1 * (imageHeight - 1) + y * (heightScale) :
                0.5 * (y1 + y2) * (imageHeight - 1);
            if (yInd < 0 || yInd > imageHeight - 1) {
                for (let x = 0; x < cropWidth; x++) {
                    for (let c = 0; c < numChannels; c++) {
                        const ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                        output.values[ind] = extrapolationValue;
                    }
                }
                continue;
            }
            if (method === 'bilinear') {
                const topInd = Math.floor(yInd);
                const bottomInd = Math.ceil(yInd);
                const yLerp = yInd - topInd;
                for (let x = 0; x < cropWidth; x++) {
                    const xInd = (cropWidth > 1) ?
                        x1 * (imageWidth - 1) + x * widthScale :
                        0.5 * (x1 + x2) * (imageWidth - 1);
                    if (xInd < 0 || xInd > imageWidth - 1) {
                        for (let c = 0; c < numChannels; c++) {
                            const ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                            output.values[ind] = extrapolationValue;
                        }
                        continue;
                    }
                    const leftInd = Math.floor(xInd);
                    const rightInd = Math.ceil(xInd);
                    const xLerp = xInd - leftInd;
                    for (let c = 0; c < numChannels; c++) {
                        let ind = c + leftInd * inStride[2] + topInd * inStride[1] +
                            bInd * inStride[0];
                        const topLeft = imageVals[ind];
                        ind = c + rightInd * inStride[2] + topInd * inStride[1] +
                            bInd * inStride[0];
                        const topRight = imageVals[ind];
                        ind = c + leftInd * inStride[2] + bottomInd * inStride[1] +
                            bInd * inStride[0];
                        const bottomLeft = imageVals[ind];
                        ind = c + rightInd * inStride[2] + bottomInd * inStride[1] +
                            bInd * inStride[0];
                        const bottomRight = imageVals[ind];
                        const top = topLeft + (topRight - topLeft) * xLerp;
                        const bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
                        ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                        output.values[ind] = top + ((bottom - top) * yLerp);
                    }
                }
            }
            else { // method == "nearest"
                for (let x = 0; x < cropWidth; ++x) {
                    const xInd = (cropWidth > 1) ?
                        x1 * (imageWidth - 1) + x * widthScale :
                        0.5 * (x1 + x2) * (imageWidth - 1);
                    if (xInd < 0 || xInd > imageWidth - 1) {
                        for (let c = 0; c < numChannels; c++) {
                            const ind = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                            output.values[ind] = extrapolationValue;
                        }
                        continue;
                    }
                    const closestX = Math.round(xInd);
                    const closestY = Math.round(yInd);
                    for (let c = 0; c < numChannels; c++) {
                        const inInd = c + closestX * inStride[2] + closestY * inStride[1] +
                            bInd * inStride[0];
                        const outInd = c + x * outStride[2] + y * outStride[1] + b * outStride[0];
                        output.values[outInd] = imageVals[inInd];
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(output.shape, output.dtype, output.values);
}
const cropAndResizeConfig = {
    kernelName: CropAndResize,
    backendName: 'cpu',
    kernelFunc: cropAndResize
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function cumsum(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, exclusive, reverse } = attrs;
    assertNotComplex(x, 'cumsum');
    const permutation = getAxesPermutation([axis], x.shape.length);
    let $x = x;
    if (permutation != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutation } });
    }
    const permutedAxis = getInnerMostAxes(1, x.shape.length)[0];
    if (permutedAxis !== $x.shape.length - 1) {
        throw new Error(`backend.cumsum in CPU expects an inner-most ` +
            `axis=${$x.shape.length - 1} but got axis=${permutedAxis}`);
    }
    const resultDtype = upcastType($x.dtype, 'int32');
    const vals = makeZerosTypedArray(sizeFromShape($x.shape), resultDtype);
    const aVals = backend.data.get($x.dataId).values;
    const finalDim = $x.shape[$x.shape.length - 1];
    const indexAdjuster = reverse ?
        (i, j) => i + finalDim - j - 1 :
        (i, j) => i + j;
    for (let i = 0; i < aVals.length; i += finalDim) {
        for (let j = 0; j < finalDim; j++) {
            const idx = indexAdjuster(i, j);
            if (j === 0) {
                vals[idx] = exclusive ? 0 : aVals[idx];
            }
            else {
                const prevIdx = indexAdjuster(i, j - 1);
                vals[idx] = exclusive ? aVals[prevIdx] + vals[prevIdx] :
                    aVals[idx] + vals[prevIdx];
            }
        }
    }
    const result = backend.makeTensorInfo($x.shape, resultDtype, vals);
    if (permutation != null) {
        const reversePermutation = getUndoAxesPermutation(permutation);
        const reverseTransposedResult = transpose({ inputs: { x: result }, backend, attrs: { perm: reversePermutation } });
        backend.disposeIntermediateTensorInfo(result);
        backend.disposeIntermediateTensorInfo($x);
        return reverseTransposedResult;
    }
    return result;
}
const cumsumConfig = {
    kernelName: Cumsum,
    backendName: 'cpu',
    kernelFunc: cumsum
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function denseBincount(args) {
    const { inputs, backend, attrs } = args;
    const { x, weights } = inputs;
    const { size, binaryOutput } = attrs;
    if (x.shape.length === 1) {
        const xVals = backend.data.get(x.dataId).values;
        const weightsVals = backend.data.get(weights.dataId).values;
        const outVals = bincountImpl(xVals, weightsVals, weights.dtype, weights.shape, size);
        return backend.makeTensorInfo([size], weights.dtype, outVals);
    }
    else if (x.shape.length === 2) {
        const xBuf = backend.bufferSync(x);
        const weightsBuf = backend.bufferSync(weights);
        const outBuf = bincountReduceImpl(xBuf, weightsBuf, size, binaryOutput);
        return backend.makeTensorInfo(outBuf.shape, weights.dtype, outBuf.values);
    }
    throw new Error(`Error in denseBincount: input must be at most rank 2, but got rank` +
        `${x.shape.length}.`);
}
const denseBincountConfig = {
    kernelName: DenseBincount,
    backendName: 'cpu',
    kernelFunc: denseBincount
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function depthToSpace(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockSize, dataFormat } = attrs;
    assert(dataFormat === 'NHWC', () => `Only NHWC dataFormat supported on CPU for depthToSpace. Got ${dataFormat}`);
    assert(blockSize > 1, () => `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);
    const batchSize = x.shape[0];
    const inputHeight = x.shape[1];
    const inputWidth = x.shape[2];
    const inputDepth = x.shape[3];
    const outputHeight = inputHeight * blockSize;
    const outputWidth = inputWidth * blockSize;
    const outputDepth = inputDepth / (blockSize * blockSize);
    const xValues = backend.data.get(x.dataId).values;
    const result = new Float32Array(batchSize * outputHeight * outputWidth * outputDepth);
    let outputIdx = 0;
    for (let b = 0; b < batchSize; ++b) {
        for (let h = 0; h < outputHeight; ++h) {
            const inH = Math.floor(h / blockSize);
            const offsetH = (h % blockSize);
            for (let w = 0; w < outputWidth; ++w) {
                const inW = Math.floor(w / blockSize);
                const offsetW = (w % blockSize);
                const offsetD = (offsetH * blockSize + offsetW) * outputDepth;
                for (let d = 0; d < outputDepth; ++d) {
                    const inD = d + offsetD;
                    const inputIdx = inD + inputDepth * (inW + inputWidth * (inH + inputHeight * b));
                    result[outputIdx++] = xValues[inputIdx];
                }
            }
        }
    }
    return backend.makeTensorInfo([batchSize, outputHeight, outputWidth, outputDepth], x.dtype, result);
}
const depthToSpaceConfig = {
    kernelName: DepthToSpace,
    backendName: 'cpu',
    kernelFunc: depthToSpace
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function depthwiseConv2dNative(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter } = inputs;
    const { strides, pad, dilations, dimRoundingMode } = attrs;
    assertNotComplex([x, filter], 'depthwiseConv2DNative');
    const xStrides = computeStrides(x.shape);
    const filterStrides = computeStrides(filter.shape);
    let $dilations = dilations;
    if ($dilations == null) {
        $dilations = [1, 1];
    }
    assert(eitherStridesOrDilationsAreOne(strides, $dilations), () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
        `1. Got strides ${strides} and dilations '${$dilations}'`);
    const convInfo = computeConv2DInfo(x.shape, filter.shape, strides, $dilations, pad, dimRoundingMode, true /* depthwise */);
    const { filterHeight, filterWidth, dilationHeight, dilationWidth, padInfo } = convInfo;
    const padLeft = padInfo.left;
    const padTop = padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;
    const y = new TensorBuffer(convInfo.outShape, x.dtype);
    const xVals = backend.data.get(x.dataId).values;
    const wVals = backend.data.get(filter.dataId).values;
    const yVals = y.values;
    for (let b = 0; b < convInfo.batchSize; ++b) {
        const xOffset1 = b * xStrides[0];
        const yOffset1 = b * y.strides[0];
        for (let yR = 0; yR < convInfo.outHeight; ++yR) {
            const yOffset2 = yOffset1 + yR * y.strides[1];
            const xRCorner = yR * convInfo.strideHeight - padLeft;
            for (let wR = 0; wR < filterHeight; ++wR) {
                const xR = xRCorner + wR * dilationHeight;
                if (xR < 0 || xR >= convInfo.inHeight) {
                    continue;
                }
                const wOffset1 = wR * filterStrides[0];
                const xOffset2 = xOffset1 + xR * xStrides[1];
                for (let yC = 0; yC < convInfo.outWidth; ++yC) {
                    const yOffset3 = yOffset2 + yC * y.strides[2];
                    const xCCorner = yC * convInfo.strideWidth - padTop;
                    for (let wC = 0; wC < filterWidth; ++wC) {
                        const xC = xCCorner + wC * dilationWidth;
                        if (xC < 0 || xC >= convInfo.inWidth) {
                            continue;
                        }
                        const wOffset2 = wOffset1 + wC * filterStrides[1];
                        const xOffset3 = xOffset2 + xC * convInfo.inChannels;
                        let yOffset4 = yOffset3;
                        let wOffset3 = wOffset2;
                        for (let d1 = 0; d1 < convInfo.inChannels; ++d1) {
                            const xVal = xVals[xOffset3 + d1];
                            for (let q = 0; q < chMul; ++q) {
                                yVals[yOffset4 + q] += xVal * wVals[wOffset3 + q];
                            }
                            yOffset4 += chMul;
                            wOffset3 += chMul;
                        }
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(y.shape, y.dtype, y.values);
}
const depthwiseConv2dNativeConfig = {
    kernelName: DepthwiseConv2dNative,
    backendName: 'cpu',
    kernelFunc: depthwiseConv2dNative
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function depthwiseConv2dNativeBackpropFilter(args) {
    const { inputs, backend, attrs } = args;
    const { x, dy } = inputs;
    const { strides, dilations, pad, dimRoundingMode, filterShape } = attrs;
    assertNotComplex([x, dy], 'depthwiseConv2dNativeBackpropFilter');
    const convInfo = computeConv2DInfo(x.shape, filterShape, strides, dilations, pad, dimRoundingMode, true /* depthwise */);
    const { strideHeight, strideWidth, filterHeight, filterWidth } = convInfo;
    const dW = new TensorBuffer(convInfo.filterShape, 'float32');
    const leftPad = convInfo.padInfo.left;
    const topPad = convInfo.padInfo.top;
    const chMul = convInfo.outChannels / convInfo.inChannels;
    const xVals = backend.data.get(x.dataId).values;
    const xBuf = new TensorBuffer(x.shape, x.dtype, xVals);
    const dyVals = backend.data.get(dy.dataId).values;
    const dyBuf = new TensorBuffer(dy.shape, dy.dtype, dyVals);
    for (let wR = 0; wR < filterHeight; ++wR) {
        const yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
        const yRMax = Math.min(convInfo.outHeight, (convInfo.inHeight + topPad - wR) / strideHeight);
        for (let wC = 0; wC < filterWidth; ++wC) {
            const yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
            const yCMax = Math.min(convInfo.outWidth, (convInfo.inWidth + leftPad - wC) / strideWidth);
            for (let d2 = 0; d2 < convInfo.outChannels; ++d2) {
                const d1 = Math.trunc(d2 / chMul);
                const dm = d2 % chMul;
                let dotProd = 0;
                for (let b = 0; b < convInfo.batchSize; ++b) {
                    for (let yR = yRMin; yR < yRMax; ++yR) {
                        const xR = wR + yR * strideHeight - topPad;
                        for (let yC = yCMin; yC < yCMax; ++yC) {
                            const xC = wC + yC * strideWidth - leftPad;
                            dotProd += xBuf.get(b, xR, xC, d1) *
                                dyBuf.get(b, yR, yC, d2);
                        }
                    }
                }
                dW.set(dotProd, wR, wC, d1, dm);
            }
        }
    }
    return backend.makeTensorInfo(dW.shape, dW.dtype, dW.values);
}
const depthwiseConv2dNativeBackpropFilterConfig = {
    kernelName: DepthwiseConv2dNativeBackpropFilter,
    backendName: 'cpu',
    kernelFunc: depthwiseConv2dNativeBackpropFilter
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function depthwiseConv2dNativeBackpropInput(args) {
    const { inputs, backend, attrs } = args;
    const { dy, filter } = inputs;
    const { strides, dilations, pad, dimRoundingMode, inputShape } = attrs;
    assertNotComplex([dy, filter], 'depthwiseConv2DNativeBackpropInput');
    const dyStrides = computeStrides(dy.shape);
    const filterStrides = computeStrides(filter.shape);
    const convInfo = computeConv2DInfo(inputShape, filter.shape, strides, dilations, pad, dimRoundingMode, true /* depthwise */);
    const dx = new TensorBuffer(convInfo.inShape, 'float32');
    const dxValues = dx.values;
    const [dxS0, dxS1, dxS2] = dx.strides;
    const dyValues = backend.data.get(dy.dataId).values;
    const [dyS0, dyS1, dyS2] = dyStrides;
    const fltValues = backend.data.get(filter.dataId).values;
    const [fltS0, fltS1, fltS2] = filterStrides;
    const { batchSize, filterHeight, filterWidth, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, strideHeight, strideWidth } = convInfo;
    const topPad = filterHeight - 1 - convInfo.padInfo.top;
    const leftPad = filterWidth - 1 - convInfo.padInfo.left;
    const chMul = outChannels / inChannels;
    for (let b = 0; b < batchSize; ++b) {
        for (let d1 = 0; d1 < inChannels; ++d1) {
            for (let xR = 0; xR < inHeight; ++xR) {
                const xRCorner = xR - topPad;
                const xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                const yRMax = Math.min(outHeight, (filterHeight + xRCorner) / strideHeight);
                for (let xC = 0; xC < inWidth; ++xC) {
                    const xCCorner = xC - leftPad;
                    const xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                    const yCMax = Math.min(outWidth, (filterWidth + xCCorner) / strideWidth);
                    let dotProd = 0;
                    for (let yR = xRMin; yR < yRMax; ++yR) {
                        const wR = yR * strideHeight - xRCorner;
                        for (let yC = xCMin; yC < yCMax; ++yC) {
                            const wC = yC * strideWidth - xCCorner;
                            const dyOffset = dyS0 * b + dyS1 * yR + dyS2 * yC;
                            const fltOffset = fltS0 * (filterHeight - 1 - wR) +
                                fltS1 * (filterWidth - 1 - wC) + fltS2 * d1;
                            for (let dm = 0; dm < chMul; ++dm) {
                                const d2 = d1 * chMul + dm;
                                const pixel = dyValues[dyOffset + d2];
                                const weight = fltValues[fltOffset + dm];
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    dxValues[dxS0 * b + dxS1 * xR + dxS2 * xC + d1] = dotProd;
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
const depthwiseConv2dNativeBackpropInputConfig = {
    kernelName: DepthwiseConv2dNativeBackpropInput,
    backendName: 'cpu',
    kernelFunc: depthwiseConv2dNativeBackpropInput
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function diag(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    const xSize = sizeFromShape(x.shape);
    const xVals = backend.data.get(x.dataId).values;
    const outBuf = buffer([xSize, xSize], x.dtype);
    const vals = outBuf.values;
    for (let i = 0; i < xVals.length; i++) {
        vals[i * xSize + i] = xVals[i];
    }
    const outShape = [...x.shape, ...x.shape];
    return backend.makeTensorInfo(outShape, outBuf.dtype, outBuf.values);
}
const diagConfig = {
    kernelName: Diag,
    backendName: 'cpu',
    kernelFunc: diag
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const dilation2dConfig = {
    kernelName: Dilation2D,
    backendName: 'cpu',
    kernelFunc: ({ inputs, backend, attrs }) => {
        const { x, filter } = inputs;
        const { strides, pad, dilations } = attrs;
        const cpuBackend = backend;
        const xVals = cpuBackend.data.get(x.dataId).values;
        const xRank = x.shape.length;
        const filterVals = cpuBackend.data.get(filter.dataId).values;
        const filterRank = filter.shape.length;
        const { batchSize, inHeight, inWidth, inChannels, outHeight, outWidth, padInfo, strideHeight, strideWidth, filterHeight, filterWidth, dilationHeight, dilationWidth, outShape } = computeDilation2DInfo(x.shape, filter.shape, strides, pad, 'NHWC' /* dataFormat */, dilations);
        const outSize = sizeFromShape(outShape);
        const outRank = outShape.length;
        const outputVals = getArrayFromDType(x.dtype, outSize);
        // Upsampling the input by fill in `dilation size - 1` values between each
        // input value.
        // This implementation follows the TF c++ implementation:
        // https://github.com/tensorflow/tensorflow/blob/d9a3a849edc198e90172bc58eb293de457f9d986/tensorflow/core/kernels/dilation_ops.cc
        for (let b = 0; b < batchSize; ++b) {
            for (let hOut = 0; hOut < outHeight; ++hOut) {
                const hBeg = hOut * strideHeight - padInfo.top;
                for (let wOut = 0; wOut < outWidth; ++wOut) {
                    const wBeg = wOut * strideWidth - padInfo.left;
                    for (let d = 0; d < inChannels; ++d) {
                        let curVal = Number.MIN_SAFE_INTEGER;
                        for (let h = 0; h < filterHeight; ++h) {
                            const hIn = hBeg + h * dilationHeight;
                            if (hIn >= 0 && hIn < inHeight) {
                                for (let w = 0; w < filterWidth; ++w) {
                                    const wIn = wBeg + w * dilationWidth;
                                    if (wIn >= 0 && wIn < inWidth) {
                                        const xIndex = locToIndex([b, hIn, wIn, d], xRank, computeStrides(x.shape));
                                        const filterIndex = locToIndex([h, w, d], filterRank, computeStrides(filter.shape));
                                        const val = xVals[xIndex] + filterVals[filterIndex];
                                        if (val > curVal) {
                                            curVal = val;
                                        }
                                    }
                                }
                            }
                        }
                        const outputIndex = locToIndex([b, hOut, wOut, d], outRank, computeStrides(outShape));
                        outputVals[outputIndex] = curVal;
                    }
                }
            }
        }
        const dataId = cpuBackend.write(toTypedArray(outputVals, x.dtype), outShape, x.dtype);
        return { dataId, shape: outShape, dtype: x.dtype };
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const dilation2dBackpropFilterConfig = {
    kernelName: Dilation2DBackpropFilter,
    backendName: 'cpu',
    kernelFunc: ({ inputs, backend, attrs }) => {
        const { x, filter, dy } = inputs;
        const { strides, pad, dilations } = attrs;
        const cpuBackend = backend;
        const $x = toNestedArray(x.shape, cpuBackend.data.get(x.dataId).values);
        const $filter = toNestedArray(filter.shape, cpuBackend.data.get(filter.dataId).values);
        const { batchSize, inHeight, inWidth, inChannels, outHeight, outWidth, padInfo, strideHeight, strideWidth, filterHeight, filterWidth, dilationHeight, dilationWidth, outShape } = computeDilation2DInfo(x.shape, filter.shape, strides, pad, 'NHWC' /* dataFormat */, dilations);
        assert(dy.rank === outShape.length, () => `Error in ${Dilation2DBackpropFilter}, dy ` +
            `must have the same rank as output ${outShape.length}, but got ` +
            `${dy.rank}`);
        const $dy = toNestedArray(outShape, cpuBackend.data.get(dy.dataId).values);
        // The computed filter gradients has the same dimensions as the filter:
        // [filterHeight, filterWidth, depth]
        const gradients = makeZerosNestedTypedArray(filter.shape, filter.dtype);
        // In the case of multiple argmax branches, we only back-propagate along the
        // last branch, i.e., the one with largest value of `h * filter_cols + w`,
        // similarly to the max-pooling backward routines.
        // This implementation follows the TF c++ implementation:
        // https://github.com/tensorflow/tensorflow/blob/d9a3a849edc198e90172bc58eb293de457f9d986/tensorflow/core/kernels/dilation_ops.cc
        for (let b = 0; b < batchSize; ++b) {
            for (let hOut = 0; hOut < outHeight; ++hOut) {
                const hBeg = hOut * strideHeight - padInfo.top;
                for (let wOut = 0; wOut < outWidth; ++wOut) {
                    const wBeg = wOut * strideWidth - padInfo.left;
                    for (let d = 0; d < inChannels; ++d) {
                        let curVal = Number.MIN_SAFE_INTEGER;
                        let hMax = 0;
                        let wMax = 0;
                        for (let h = 0; h < filterHeight; ++h) {
                            const hIn = hBeg + h * dilationHeight;
                            if (hIn >= 0 && hIn < inHeight) {
                                for (let w = 0; w < filterWidth; ++w) {
                                    const wIn = wBeg + w * dilationWidth;
                                    if (wIn >= 0 && wIn < inWidth) {
                                        const val = $x[b][hIn][wIn][d] + $filter[h][w][d];
                                        if (val > curVal) {
                                            curVal = val;
                                            hMax = h;
                                            wMax = w;
                                        }
                                    }
                                }
                            }
                        }
                        gradients[hMax][wMax][d] += $dy[b][hOut][wOut][d];
                    }
                }
            }
        }
        const dataId = cpuBackend.write(toTypedArray(gradients, x.dtype), filter.shape, filter.dtype);
        return { dataId, shape: filter.shape, dtype: filter.dtype };
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const dilation2dBackpropInputConfig = {
    kernelName: Dilation2DBackpropInput,
    backendName: 'cpu',
    kernelFunc: ({ inputs, backend, attrs }) => {
        const { x, filter, dy } = inputs;
        const { strides, pad, dilations } = attrs;
        const cpuBackend = backend;
        const $x = toNestedArray(x.shape, cpuBackend.data.get(x.dataId).values);
        const $filter = toNestedArray(filter.shape, cpuBackend.data.get(filter.dataId).values);
        const { batchSize, inHeight, inWidth, inChannels, outHeight, outWidth, padInfo, strideHeight, strideWidth, filterHeight, filterWidth, dilationHeight, dilationWidth, outShape } = computeDilation2DInfo(x.shape, filter.shape, strides, pad, 'NHWC' /* dataFormat */, dilations);
        assert(dy.rank === outShape.length, () => `Error in ${Dilation2DBackpropInput}, dy ` +
            `must have the same rank as output ${outShape.length}, but got ` +
            `${dy.rank}`);
        const $dy = toNestedArray(outShape, cpuBackend.data.get(dy.dataId).values);
        // The computed gradients has the same dimensions as the input:
        // [batch, inputHeight, inputCols, inChannel]
        const gradients = makeZerosNestedTypedArray(x.shape, x.dtype);
        // In the case of multiple argmax branches, we only back-propagate along the
        // last branch, i.e., the one with largest value of `h * filter_cols + w`,
        // similarly to the max-pooling backward routines.
        // This implementation follows the TF c++ implementation:
        // https://github.com/tensorflow/tensorflow/blob/d9a3a849edc198e90172bc58eb293de457f9d986/tensorflow/core/kernels/dilation_ops.cc
        for (let b = 0; b < batchSize; ++b) {
            for (let hOut = 0; hOut < outHeight; ++hOut) {
                const hBeg = hOut * strideHeight - padInfo.top;
                for (let wOut = 0; wOut < outWidth; ++wOut) {
                    const wBeg = wOut * strideWidth - padInfo.left;
                    for (let d = 0; d < inChannels; ++d) {
                        let curVal = Number.MIN_SAFE_INTEGER;
                        let hInMax = (hBeg < 0) ? 0 : hBeg;
                        let wInMax = (wBeg < 0) ? 0 : wBeg;
                        for (let h = 0; h < filterHeight; ++h) {
                            const hIn = hBeg + h * dilationHeight;
                            if (hIn >= 0 && hIn < inHeight) {
                                for (let w = 0; w < filterWidth; ++w) {
                                    const wIn = wBeg + w * dilationWidth;
                                    if (wIn >= 0 && wIn < inWidth) {
                                        const val = $x[b][hIn][wIn][d] + $filter[h][w][d];
                                        if (val > curVal) {
                                            curVal = val;
                                            hInMax = hIn;
                                            wInMax = wIn;
                                        }
                                    }
                                }
                            }
                        }
                        gradients[b][hInMax][wInMax][d] += $dy[b][hOut][wOut][d];
                    }
                }
            }
        }
        const dataId = cpuBackend.write(toTypedArray(gradients, x.dtype), x.shape, x.dtype);
        return { dataId, shape: x.shape, dtype: x.dtype };
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function eluGrad(args) {
    const { inputs, backend } = args;
    const { dy, y } = inputs;
    assertNotComplex([dy, y], 'eluGrad');
    const resultValues = new Float32Array(sizeFromShape(y.shape));
    const values = backend.data.get(y.dataId).values;
    const dyValues = backend.data.get(dy.dataId).values;
    for (let i = 0; i < values.length; ++i) {
        const v = values[i];
        if (v >= 1) {
            resultValues[i] = dyValues[i];
        }
        else {
            resultValues[i] = dyValues[i] * (v + 1);
        }
    }
    return backend.makeTensorInfo(y.shape, 'float32', resultValues);
}
const eluGradConfig = {
    kernelName: EluGrad,
    backendName: 'cpu',
    kernelFunc: eluGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const equalImpl = createSimpleBinaryKernelImpl((a, b) => (a === b) ? 1 : 0);
const equal = binaryKernelFunc(Equal, equalImpl, null /* complexImpl */, 'bool');
const equalConfig = {
    kernelName: Equal,
    backendName: 'cpu',
    kernelFunc: equal
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const p = ERF_P;
const a1 = ERF_A1;
const a2 = ERF_A2;
const a3 = ERF_A3;
const a4 = ERF_A4;
const a5 = ERF_A5;
const erf = unaryKernelFunc(Erf, (xi) => {
    const sign = Math.sign(xi);
    const v = Math.abs(xi);
    const t = 1.0 / (1.0 + p * v);
    return sign *
        (1.0 -
            (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                Math.exp(-v * v));
});
const erfConfig = {
    kernelName: Erf,
    backendName: 'cpu',
    kernelFunc: erf,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function expandDims(args) {
    const { inputs, backend, attrs } = args;
    const { input } = inputs;
    const { dim } = attrs;
    const inputRank = input.shape.length;
    const newShape = input.shape.slice();
    let $dim = dim;
    if (dim < 0) {
        // Negative value is counted from the tail of rank.
        assert(-(inputRank + 1) <= dim, () => `Axis must be in the interval [${-(inputRank + 1)}, ${inputRank}]`);
        $dim = inputRank + dim + 1;
    }
    newShape.splice($dim, 0, 1);
    return reshape({ inputs: { x: input }, backend, attrs: { shape: newShape } });
}
const expandDimsConfig = {
    kernelName: ExpandDims,
    backendName: 'cpu',
    kernelFunc: expandDims
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const realDivImpl = createSimpleBinaryKernelImpl((a, b) => a / b);
const div = binaryKernelFunc(RealDiv, realDivImpl);
const realDivConfig = {
    kernelName: RealDiv,
    backendName: 'cpu',
    kernelFunc: div
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
/**
 * Calculate FFT of inner most elements of batch tensor.
 */
function fftBatch(input, inverse, cpuBackend) {
    const inputShape = input.shape;
    const batch = inputShape[0];
    const innerDim = inputShape[1];
    const inputVals = cpuBackend.data.get(input.dataId);
    const real2D = inputVals.complexTensorInfos.real;
    const imag2D = inputVals.complexTensorInfos.imag;
    // Collects real and imaginary values separately.
    const resultShape = [batch, innerDim];
    const resultSize = sizeFromShape(resultShape);
    const resultReal = getTypedArrayFromDType('float32', resultSize);
    const resultImag = getTypedArrayFromDType('float32', resultSize);
    for (let b = 0; b < batch; b++) {
        // TODO: Support slice ops for complex type.
        const r = slice({
            inputs: { x: real2D },
            backend: cpuBackend,
            attrs: { begin: [b, 0], size: [1, innerDim] }
        });
        const i = slice({
            inputs: { x: imag2D },
            backend: cpuBackend,
            attrs: { begin: [b, 0], size: [1, innerDim] }
        });
        const input = complex({ inputs: { real: r, imag: i }, backend: cpuBackend });
        // Run FFT by batch element.
        const { real, imag } = fftImpl(input, inverse, cpuBackend);
        const res = mergeRealAndImagArrays(real, imag);
        for (let d = 0; d < innerDim; d++) {
            const c = getComplexWithIndex(res, d);
            resultReal[b * innerDim + d] = c.real;
            resultImag[b * innerDim + d] = c.imag;
        }
        cpuBackend.disposeIntermediateTensorInfo(r);
        cpuBackend.disposeIntermediateTensorInfo(i);
        cpuBackend.disposeIntermediateTensorInfo(input);
    }
    const $realInfo = cpuBackend.makeTensorInfo(resultShape, 'float32', resultReal);
    const $imagInfo = cpuBackend.makeTensorInfo(resultShape, 'float32', resultImag);
    const result = complex({ inputs: { real: $realInfo, imag: $imagInfo }, backend: cpuBackend });
    cpuBackend.disposeIntermediateTensorInfo($realInfo);
    cpuBackend.disposeIntermediateTensorInfo($imagInfo);
    return result;
}
function fftImpl(input, inverse, cpuBackend) {
    const inputSize = sizeFromShape(input.shape);
    const inputVals = cpuBackend.data.get(input.dataId);
    const realVals = cpuBackend.data.get(inputVals.complexTensorInfos.real.dataId).values;
    const imagVals = cpuBackend.data.get(inputVals.complexTensorInfos.imag.dataId).values;
    if (isExponentOf2(inputSize)) {
        const result = fftRadix2(realVals, imagVals, inputSize, inverse, cpuBackend);
        const resultShape = [input.shape[0], input.shape[1]];
        if (inverse) {
            const realInfo = cpuBackend.makeTensorInfo(resultShape, 'float32', result.real);
            const imagInfo = cpuBackend.makeTensorInfo(resultShape, 'float32', result.imag);
            const sizeInfo = cpuBackend.makeTensorInfo([], 'float32', createScalarValue(inputSize, 'float32'));
            const sizeInfoCopy = identity({ inputs: { x: sizeInfo }, backend: cpuBackend });
            const divRealInfo = realDivConfig.kernelFunc({ inputs: { a: realInfo, b: sizeInfo }, backend: cpuBackend });
            const divImagInfo = realDivConfig.kernelFunc({ inputs: { a: imagInfo, b: sizeInfoCopy }, backend: cpuBackend });
            const divRealVals = cpuBackend.data.get(divRealInfo.dataId).values;
            const divImagVals = cpuBackend.data.get(divImagInfo.dataId).values;
            cpuBackend.disposeIntermediateTensorInfo(realInfo);
            cpuBackend.disposeIntermediateTensorInfo(imagInfo);
            cpuBackend.disposeIntermediateTensorInfo(sizeInfo);
            cpuBackend.disposeIntermediateTensorInfo(sizeInfoCopy);
            cpuBackend.disposeIntermediateTensorInfo(divRealInfo);
            cpuBackend.disposeIntermediateTensorInfo(divImagInfo);
            return { real: divRealVals, imag: divImagVals };
        }
        return result;
    }
    else {
        const data = mergeRealAndImagArrays(realVals, imagVals);
        const rawOutput = fourierTransformByMatmul(data, inputSize, inverse);
        return splitRealAndImagArrays(rawOutput);
    }
}
function isExponentOf2(size) {
    return (size & size - 1) === 0;
}
// FFT using Cooley-Tukey algorithm on radix 2 dimensional input.
function fftRadix2(realVals, imagVals, size, inverse, cpuBackend) {
    if (size === 1) {
        return { real: realVals, imag: imagVals };
    }
    const data = mergeRealAndImagArrays(realVals, imagVals);
    const half = size / 2;
    const evenComplex = complexWithEvenIndex(data);
    const evenRealVals = evenComplex.real;
    const evenImagVals = evenComplex.imag;
    const evenShape = [evenRealVals.length];
    const evenRealInfo = cpuBackend.makeTensorInfo(evenShape, 'float32', evenRealVals);
    const evenImagInfo = cpuBackend.makeTensorInfo(evenShape, 'float32', evenImagVals);
    const evenTensorInfo = complex({ inputs: { real: evenRealInfo, imag: evenImagInfo }, backend: cpuBackend });
    const oddComplex = complexWithOddIndex(data);
    const oddRealVals = oddComplex.real;
    const oddImagVals = oddComplex.imag;
    const oddShape = [oddRealVals.length];
    const oddRealInfo = cpuBackend.makeTensorInfo(oddShape, 'float32', oddRealVals);
    const oddImagInfo = cpuBackend.makeTensorInfo(oddShape, 'float32', oddImagVals);
    const oddTensorInfo = complex({ inputs: { real: oddRealInfo, imag: oddImagInfo }, backend: cpuBackend });
    // Recursive call for half part of original input.
    const $evenComplex = fftRadix2(evenRealVals, evenImagVals, half, inverse, cpuBackend);
    const $evenRealVals = $evenComplex.real;
    const $evenImagVals = $evenComplex.imag;
    const $evenShape = [$evenRealVals.length];
    const $evenRealInfo = cpuBackend.makeTensorInfo($evenShape, 'float32', $evenRealVals);
    const $evenImagInfo = cpuBackend.makeTensorInfo($evenShape, 'float32', $evenImagVals);
    const $evenTensorInfo = complex({
        inputs: { real: $evenRealInfo, imag: $evenImagInfo },
        backend: cpuBackend
    });
    const $oddComplex = fftRadix2(oddRealVals, oddImagVals, half, inverse, cpuBackend);
    const $oddRealVals = $oddComplex.real;
    const $oddImagVals = $oddComplex.imag;
    const $oddShape = [$oddRealVals.length];
    const $oddRealInfo = cpuBackend.makeTensorInfo($oddShape, 'float32', $oddRealVals);
    const $oddImagInfo = cpuBackend.makeTensorInfo($oddShape, 'float32', $oddImagVals);
    const $oddTensorInfo = complex({ inputs: { real: $oddRealInfo, imag: $oddImagInfo }, backend: cpuBackend });
    const e = exponents(size, inverse);
    const eShape = [e.real.length];
    const eRealInfo = cpuBackend.makeTensorInfo(eShape, 'float32', e.real);
    const eImagInfo = cpuBackend.makeTensorInfo(eShape, 'float32', e.imag);
    const complexInfo = complex({ inputs: { real: eRealInfo, imag: eImagInfo }, backend: cpuBackend });
    const exponentInfo = multiply({ inputs: { a: complexInfo, b: $oddTensorInfo }, backend: cpuBackend });
    const addPart = add({
        inputs: { a: $evenTensorInfo, b: exponentInfo },
        backend: cpuBackend
    });
    const subPart = sub({
        inputs: { a: $evenTensorInfo, b: exponentInfo },
        backend: cpuBackend
    });
    const addPartReal = real({ inputs: { input: addPart }, backend: cpuBackend });
    const subPartReal = real({ inputs: { input: subPart }, backend: cpuBackend });
    const addPartImag = imag({ inputs: { input: addPart }, backend: cpuBackend });
    const subPartImag = imag({ inputs: { input: subPart }, backend: cpuBackend });
    const $real = concat({
        inputs: [addPartReal, subPartReal],
        backend: cpuBackend,
        attrs: { axis: 0 }
    });
    const $imag = concat({
        inputs: [addPartImag, subPartImag],
        backend: cpuBackend,
        attrs: { axis: 0 }
    });
    const $realVals = cpuBackend.data.get($real.dataId).values;
    const $imagVals = cpuBackend.data.get($imag.dataId).values;
    cpuBackend.disposeIntermediateTensorInfo(evenRealInfo);
    cpuBackend.disposeIntermediateTensorInfo(evenImagInfo);
    cpuBackend.disposeIntermediateTensorInfo(evenTensorInfo);
    cpuBackend.disposeIntermediateTensorInfo(oddRealInfo);
    cpuBackend.disposeIntermediateTensorInfo(oddImagInfo);
    cpuBackend.disposeIntermediateTensorInfo(oddTensorInfo);
    cpuBackend.disposeIntermediateTensorInfo($evenRealInfo);
    cpuBackend.disposeIntermediateTensorInfo($evenImagInfo);
    cpuBackend.disposeIntermediateTensorInfo($evenTensorInfo);
    cpuBackend.disposeIntermediateTensorInfo($oddRealInfo);
    cpuBackend.disposeIntermediateTensorInfo($oddImagInfo);
    cpuBackend.disposeIntermediateTensorInfo($oddTensorInfo);
    cpuBackend.disposeIntermediateTensorInfo(eRealInfo);
    cpuBackend.disposeIntermediateTensorInfo(eImagInfo);
    cpuBackend.disposeIntermediateTensorInfo(complexInfo);
    cpuBackend.disposeIntermediateTensorInfo(exponentInfo);
    cpuBackend.disposeIntermediateTensorInfo(addPart);
    cpuBackend.disposeIntermediateTensorInfo(subPart);
    cpuBackend.disposeIntermediateTensorInfo(addPartReal);
    cpuBackend.disposeIntermediateTensorInfo(addPartImag);
    cpuBackend.disposeIntermediateTensorInfo(subPartReal);
    cpuBackend.disposeIntermediateTensorInfo(subPartImag);
    cpuBackend.disposeIntermediateTensorInfo($real);
    cpuBackend.disposeIntermediateTensorInfo($imag);
    return { real: $realVals, imag: $imagVals };
}
// Calculate fourier transform by multplying sinusoid matrix.
function fourierTransformByMatmul(data, size, inverse) {
    const ret = new Float32Array(size * 2);
    // TODO: Use matmul instead once it supports complex64 type.
    for (let r = 0; r < size; r++) {
        let real = 0.0;
        let imag = 0.0;
        for (let c = 0; c < size; c++) {
            const e = exponent(r * c, size, inverse);
            const term = getComplexWithIndex(data, c);
            real += term.real * e.real - term.imag * e.imag;
            imag += term.real * e.imag + term.imag * e.real;
        }
        if (inverse) {
            real /= size;
            imag /= size;
        }
        assignToTypedArray(ret, real, imag, r);
    }
    return ret;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function fft(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const inputSize = sizeFromShape(input.shape);
    // Collapse all outer dimensions to a single batch dimension.
    const innerDimensionSize = input.shape[input.shape.length - 1];
    const batch = inputSize / innerDimensionSize;
    const input2D = reshape({
        inputs: { x: input },
        backend,
        attrs: { shape: [batch, innerDimensionSize] }
    });
    const result = fftBatch(input2D, false, backend);
    const resultReshaped = reshape({ inputs: { x: result }, backend, attrs: { shape: input.shape } });
    backend.disposeIntermediateTensorInfo(input2D);
    backend.disposeIntermediateTensorInfo(result);
    return resultReshaped;
}
const fftConfig = {
    kernelName: FFT,
    backendName: 'cpu',
    kernelFunc: fft
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function fill(args) {
    const { backend, attrs } = args;
    const { shape, value, dtype } = attrs;
    const $dtype = dtype || inferDtype(value);
    const values = getArrayFromDType($dtype, sizeFromShape(shape));
    fillValues(values, value, $dtype);
    return backend.makeTensorInfo(shape, $dtype, values);
}
const fillConfig = {
    kernelName: Fill,
    backendName: 'cpu',
    kernelFunc: fill
};
function fillValues(values, value, dtype) {
    if (dtype === 'string') {
        values.fill(value);
    }
    else {
        values.fill(value);
    }
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const flipLeftRightConfig = {
    kernelName: FlipLeftRight,
    backendName: 'cpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { image } = inputs;
        const cpuBackend = backend;
        const output = getTypedArrayFromDType(image.dtype, sizeFromShape(image.shape));
        const [batch, imageHeight, imageWidth, numChannels] = image.shape;
        const imageVals = cpuBackend.data.get(image.dataId).values;
        for (let batchIdx = 0; batchIdx < batch; batchIdx++) {
            const batchOffset = batchIdx * imageWidth * imageHeight * numChannels;
            for (let row = 0; row < imageHeight; row++) {
                const rowOffset = row * (imageWidth * numChannels);
                for (let col = 0; col < imageWidth; col++) {
                    const colOffset = col * numChannels;
                    for (let channel = 0; channel < numChannels; channel++) {
                        const coords = [batch, row, col, channel];
                        const x = coords[2];
                        const coordX = Math.round(imageWidth - x);
                        const outIdx = batchOffset + rowOffset + colOffset + channel;
                        let outputValue = imageVals[outIdx];
                        // If the coordinate position falls within the image boundaries...
                        if (coordX >= 0 && coordX < imageWidth) {
                            // set the output to the image value at the coordinate position.
                            const rotatedColOffset = coordX * numChannels;
                            const imageIdx = batchOffset + rowOffset + rotatedColOffset + channel;
                            outputValue = imageVals[imageIdx];
                        }
                        output[outIdx] = outputValue;
                    }
                }
            }
        }
        const dataId = cpuBackend.write(output, image.shape, image.dtype);
        return { dataId, shape: image.shape, dtype: image.dtype };
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const floorDivImpl = createSimpleBinaryKernelImpl((a, b) => Math.floor(a / b));
const floorDiv = binaryKernelFunc(FloorDiv, floorDivImpl, null /* complexImpl */, 'int32');
const floorDivConfig = {
    kernelName: FloorDiv,
    backendName: 'cpu',
    kernelFunc: floorDiv
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function fusedConv2D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, bias, preluActivationWeights } = inputs;
    const { strides, pad, dataFormat, dilations, dimRoundingMode, activation, leakyreluAlpha } = attrs;
    let result = conv2D({
        inputs: { x, filter },
        backend,
        attrs: { strides, pad, dataFormat, dilations, dimRoundingMode }
    });
    if (bias) {
        const resultOld = result;
        result = add({ inputs: { a: result, b: bias }, backend });
        backend.disposeIntermediateTensorInfo(resultOld);
    }
    if (activation) {
        const resultOld = result;
        result = applyActivation(backend, result, activation, preluActivationWeights, leakyreluAlpha);
        backend.disposeIntermediateTensorInfo(resultOld);
    }
    return result;
}
const fusedConv2DConfig = {
    kernelName: FusedConv2D,
    backendName: 'cpu',
    kernelFunc: fusedConv2D
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function fusedDepthwiseConv2D(args) {
    const { inputs, backend, attrs } = args;
    const { x, filter, bias, preluActivationWeights } = inputs;
    const { strides, pad, dataFormat, dilations, dimRoundingMode, activation, leakyreluAlpha } = attrs;
    let result = depthwiseConv2dNative({
        inputs: { x, filter },
        backend,
        attrs: { strides, pad, dataFormat, dilations, dimRoundingMode }
    });
    if (bias) {
        const oldResult = result;
        result = add({ inputs: { a: result, b: bias }, backend });
        backend.disposeIntermediateTensorInfo(oldResult);
    }
    if (activation) {
        const oldResult = result;
        result = applyActivation(backend, result, activation, preluActivationWeights, leakyreluAlpha);
        backend.disposeIntermediateTensorInfo(oldResult);
    }
    return result;
}
const fusedDepthwiseConv2DConfig = {
    kernelName: FusedDepthwiseConv2D,
    backendName: 'cpu',
    kernelFunc: fusedDepthwiseConv2D
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function gatherNd(args) {
    const { inputs, backend } = args;
    const { params, indices } = inputs;
    const paramsSize = sizeFromShape(params.shape);
    const indicesShape = indices.shape;
    const sliceRank = indicesShape[indicesShape.length - 1];
    const [resultShape, numSlices, sliceSize, strides] = prepareAndValidate(params, indices);
    if (numSlices === 0) {
        return backend.makeTensorInfo(resultShape, params.dtype, []);
    }
    const outBuf = buffer([numSlices, sliceSize], params.dtype);
    const indicesData = backend.data.get(indices.dataId).values;
    const paramsData = backend.data.get(params.dataId).values;
    for (let i = 0; i < numSlices; i++) {
        const index = [];
        let flattenIndex = 0;
        for (let j = 0; j < sliceRank; j++) {
            const dim = indicesData[i * sliceRank + j];
            flattenIndex += dim * strides[j];
            index.push(dim);
        }
        if (flattenIndex < 0 || flattenIndex >= paramsSize / sliceSize) {
            throw new Error(`Invalid indices: ${index} does not index into ${params.shape}`);
        }
        for (let k = 0; k < sliceSize; k++) {
            outBuf.values[i * sliceSize + k] =
                paramsData[flattenIndex * sliceSize + k];
        }
    }
    return backend.makeTensorInfo(resultShape, outBuf.dtype, outBuf.values);
}
const gatherNdConfig = {
    kernelName: GatherNd,
    backendName: 'cpu',
    kernelFunc: gatherNd
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function gatherV2(args) {
    const { inputs, backend, attrs } = args;
    const { x, indices } = inputs;
    const { axis, batchDims } = attrs;
    assertNotComplex([x, indices], 'gatherV2');
    let $batchDims = batchDims;
    if (batchDims == null) {
        $batchDims = 0;
    }
    const indicesSize = sizeFromShape(indices.shape);
    const parsedAxis = parseAxisParam(axis, x.shape)[0];
    const shapeInfo = collectGatherOpShapeInfo(x, indices, parsedAxis, $batchDims);
    const flattenX = reshape({
        inputs: { x },
        backend,
        attrs: {
            shape: [
                shapeInfo.batchSize, shapeInfo.outerSize, shapeInfo.dimSize,
                shapeInfo.sliceSize
            ]
        }
    });
    const flattenIndex = reshape({
        inputs: { x: indices },
        backend,
        attrs: { shape: [shapeInfo.batchSize, indicesSize / shapeInfo.batchSize] }
    });
    const flattenOutputShape = [
        shapeInfo.batchSize, shapeInfo.outerSize, indicesSize / shapeInfo.batchSize,
        shapeInfo.sliceSize
    ];
    const indicesBuf = backend.bufferSync(flattenIndex);
    const xBuf = backend.bufferSync(flattenX);
    const outBuf = gatherV2Impl(xBuf, indicesBuf, flattenOutputShape);
    backend.disposeIntermediateTensorInfo(flattenX);
    backend.disposeIntermediateTensorInfo(flattenIndex);
    return backend.makeTensorInfo(shapeInfo.outputShape, outBuf.dtype, outBuf.values);
}
const gatherV2Config = {
    kernelName: GatherV2,
    backendName: 'cpu',
    kernelFunc: gatherV2
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const greaterEqualImpl = createSimpleBinaryKernelImpl((a, b) => (a >= b) ? 1 : 0);
const greaterEqual = binaryKernelFunc(GreaterEqual, greaterEqualImpl, null /* complexImpl */, 'bool');
const greaterEqualConfig = {
    kernelName: GreaterEqual,
    backendName: 'cpu',
    kernelFunc: greaterEqual
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function ifft(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const inputSize = sizeFromShape(input.shape);
    // Collapse all outer dimensions to a single batch dimension.
    const innerDimensionSize = input.shape[input.shape.length - 1];
    const batch = inputSize / innerDimensionSize;
    const input2D = reshape({
        inputs: { x: input },
        backend,
        attrs: { shape: [batch, innerDimensionSize] }
    });
    const result = fftBatch(input2D, true, backend);
    const resultReshaped = reshape({ inputs: { x: result }, backend, attrs: { shape: input.shape } });
    backend.disposeIntermediateTensorInfo(input2D);
    backend.disposeIntermediateTensorInfo(result);
    return resultReshaped;
}
const ifftConfig = {
    kernelName: IFFT,
    backendName: 'cpu',
    kernelFunc: ifft
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const isFinite = unaryKernelFunc(IsFinite, (xi) => Number.isFinite(xi) ? 1 : 0, 'bool');
const isFiniteConfig = {
    kernelName: IsFinite,
    backendName: 'cpu',
    kernelFunc: isFinite,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const isInf = unaryKernelFunc(IsInf, (xi) => Math.abs(xi) === Infinity ? 1 : 0, 'bool');
const isInfConfig = {
    kernelName: IsInf,
    backendName: 'cpu',
    kernelFunc: isInf,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const isNaN$1 = unaryKernelFunc(IsNan, (xi) => Number.isNaN(xi) ? 1 : 0, 'bool');
const isNaNConfig = {
    kernelName: IsNan,
    backendName: 'cpu',
    kernelFunc: isNaN$1,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const lessEqualImpl = createSimpleBinaryKernelImpl((a, b) => (a <= b) ? 1 : 0);
const lessEqual = binaryKernelFunc(LessEqual, lessEqualImpl, null /* complexImpl */, 'bool');
const lessEqualConfig = {
    kernelName: LessEqual,
    backendName: 'cpu',
    kernelFunc: lessEqual
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function linSpace(args) {
    const { backend, attrs } = args;
    const { start, stop, num } = attrs;
    const outVals = linSpaceImpl(start, stop, num);
    return backend.makeTensorInfo([outVals.length], 'float32', outVals);
}
const linSpaceConfig = {
    kernelName: LinSpace,
    backendName: 'cpu',
    kernelFunc: linSpace
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const log1p = unaryKernelFunc(Log1p, (xi) => Math.log1p(xi));
const log1pConfig = {
    kernelName: Log1p,
    backendName: 'cpu',
    kernelFunc: log1p,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const logicalAndImpl = createSimpleBinaryKernelImpl((a, b) => a && b);
const logicalAnd = binaryKernelFunc(LogicalAnd, logicalAndImpl, null /* complexImpl */, 'bool');
const logicalAndConfig = {
    kernelName: LogicalAnd,
    backendName: 'cpu',
    kernelFunc: logicalAnd
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const logicalNot = unaryKernelFunc(LogicalNot, (xi) => xi ? 0 : 1, 'bool');
const logicalNotConfig = {
    kernelName: LogicalNot,
    backendName: 'cpu',
    kernelFunc: logicalNot,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const logicalOrImpl = createSimpleBinaryKernelImpl((a, b) => a || b);
const logicalOr = binaryKernelFunc(LogicalOr, logicalOrImpl, null /* complexImpl */, 'bool');
const logicalOrConfig = {
    kernelName: LogicalOr,
    backendName: 'cpu',
    kernelFunc: logicalOr
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function lRN(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { depthRadius, bias, alpha, beta } = attrs;
    assertNotComplex(x, 'LRN');
    const channels = x.shape[3];
    const maxD = channels - 1;
    const xValues = backend.data.get(x.dataId).values;
    const size = sizeFromShape(x.shape);
    const result = new Float32Array(size);
    function sumAcrossChannels(offset) {
        const currentChannel = offset % channels;
        let beginSumOffset = offset - currentChannel + Math.max(0, currentChannel - depthRadius);
        const endSumOffset = offset - currentChannel + Math.min(currentChannel + depthRadius, maxD);
        let sum = 0.0;
        for (; beginSumOffset <= endSumOffset; beginSumOffset++) {
            const z = xValues[beginSumOffset];
            sum += z * z;
        }
        return sum;
    }
    for (let offset = 0; offset < size; offset++) {
        const sum = sumAcrossChannels(offset);
        const val = xValues[offset] * Math.pow(bias + alpha * sum, -beta);
        result[offset] = val;
    }
    return backend.makeTensorInfo(x.shape, x.dtype, result);
}
const lRNConfig = {
    kernelName: LRN,
    backendName: 'cpu',
    kernelFunc: lRN
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function lRNGrad(args) {
    const { inputs, backend, attrs } = args;
    const { x, y, dy } = inputs;
    const { depthRadius, bias, alpha, beta } = attrs;
    assertNotComplex(dy, 'LRNGrad');
    const dySize = sizeFromShape(dy.shape);
    const channels = dy.shape[3];
    const dyValues = backend.data.get(dy.dataId).values;
    const xValues = backend.data.get(x.dataId).values;
    const yValues = backend.data.get(y.dataId).values;
    const result = new Float32Array(dySize);
    const size = dySize;
    for (let offset = 0; offset < size; offset++) {
        const currentChannel = offset % channels;
        const depthBegin = (offset - currentChannel) + Math.max(0, currentChannel - depthRadius);
        const depthEnd = (offset - currentChannel) +
            Math.min(channels, currentChannel + depthRadius + 1);
        let norm = 0;
        for (let k = depthBegin; k < depthEnd; k++) {
            norm += Math.pow(xValues[k], 2);
        }
        norm = alpha * norm + bias;
        for (let k = depthBegin; k < depthEnd; k++) {
            let dyi = -2 * alpha * beta * xValues[k] * yValues[offset] / norm;
            if (offset === k) {
                dyi += Math.pow(norm, -beta);
            }
            dyi *= dyValues[offset];
            result[k] += dyi;
        }
    }
    return backend.makeTensorInfo(dy.shape, x.dtype, result);
}
const lRNGradConfig = {
    kernelName: LRNGrad,
    backendName: 'cpu',
    kernelFunc: lRNGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function max(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { reductionIndices, keepDims } = attrs;
    const cpuBackend = backend;
    let xShape = x.shape;
    const xRank = xShape.length;
    const origAxes = parseAxisParam(reductionIndices, xShape);
    let axes = origAxes;
    const permutedAxes = getAxesPermutation(axes, xRank);
    let xVals = cpuBackend.data.get(x.dataId).values;
    if (permutedAxes != null) {
        const newShape = new Array(xRank);
        for (let i = 0; i < newShape.length; i++) {
            newShape[i] = xShape[permutedAxes[i]];
        }
        xVals = transposeImpl(xVals, xShape, x.dtype, permutedAxes, newShape);
        axes = getInnerMostAxes(axes.length, xRank);
        xShape = newShape;
    }
    assertNotComplex(x, 'max');
    assertAxesAreInnerMostDims('max', axes, xRank);
    const [maxOutShape, reduceShape] = computeOutAndReduceShapes(xShape, axes);
    const reduceSize = sizeFromShape(reduceShape);
    const result = maxImpl(xVals, reduceSize, maxOutShape, x.dtype);
    const dataId = cpuBackend.write(result, maxOutShape, x.dtype);
    let outShape = maxOutShape;
    if (keepDims) {
        // reshape
        const newShape = expandShapeToKeepDim(maxOutShape, origAxes);
        outShape = newShape;
    }
    return { dataId, shape: outShape, dtype: x.dtype };
}
const maxConfig = {
    kernelName: Max,
    backendName: 'cpu',
    kernelFunc: max
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function maxPool(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    assertNotComplex(x, 'maxPool');
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const dilations = 1;
    assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in maxPool: Either strides or dilations must be 1. ' +
        `Got strides ${strides} and dilations '${dilations}'`);
    const convInfo = computePool2DInfo(x.shape, filterSize, strides, dilations, pad, dimRoundingMode);
    let res;
    if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
        arraysEqual(convInfo.inShape, convInfo.outShape)) {
        res = identity({ inputs: { x }, backend });
    }
    else {
        const xValues = backend.data.get(x.dataId).values;
        const strides = computeStrides(x.shape);
        const buffer = pool(xValues, x.shape, x.dtype, strides, convInfo, 'max');
        res = backend.makeTensorInfo(convInfo.outShape, x.dtype, buffer.values);
    }
    return res;
}
const maxPoolConfig = {
    kernelName: MaxPool,
    backendName: 'cpu',
    kernelFunc: maxPool
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function maxPool3D(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { filterSize, strides, pad, dimRoundingMode, dataFormat } = attrs;
    assertNotComplex(x, 'maxPool3d');
    const convInfo = computePool3DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode, dataFormat);
    const xValues = backend.data.get(x.dataId).values;
    const outBuf = pool3d(xValues, x.shape, x.dtype, computeStrides(x.shape), convInfo, 'max');
    return backend.makeTensorInfo(outBuf.shape, 'float32', outBuf.values);
}
const maxPool3DConfig = {
    kernelName: MaxPool3D,
    backendName: 'cpu',
    kernelFunc: maxPool3D
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function maxPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    assertNotComplex([dy, input], 'maxPool3DGrad');
    const convInfo = computePool3DInfo(input.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
    const inputBuf = backend.bufferSync(input);
    const maxPosBuf = maxPool3dPositions(inputBuf, convInfo);
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = buffer(input.shape, 'float32');
    const dyBuf = backend.bufferSync(dy);
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
                for (let dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
                    for (let dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
                        // Shader code begins
                        const dyDepthCorner = dxDepth - padFront;
                        const dyRowCorner = dxRow - padTop;
                        const dyColCorner = dxCol - padLeft;
                        let dotProd = 0;
                        for (let wDepth = 0; wDepth < effectiveFilterDepth; wDepth += dilationDepth) {
                            const dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                            if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                                Math.floor(dyDepth) !== dyDepth) {
                                continue;
                            }
                            for (let wRow = 0; wRow < effectiveFilterHeight; wRow += dilationHeight) {
                                const dyRow = (dyRowCorner + wRow) / strideHeight;
                                if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                                    Math.floor(dyRow) !== dyRow) {
                                    continue;
                                }
                                for (let wCol = 0; wCol < effectiveFilterWidth; wCol += dilationWidth) {
                                    const dyCol = (dyColCorner + wCol) / strideWidth;
                                    if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                                        Math.floor(dyCol) !== dyCol) {
                                        continue;
                                    }
                                    const maxPos = effectiveFilterDepth * effectiveFilterHeight *
                                        effectiveFilterWidth -
                                        1 -
                                        maxPosBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                    const curPos = wDepth * effectiveFilterHeight * effectiveFilterWidth +
                                        wRow * effectiveFilterWidth + wCol;
                                    const mask = maxPos === curPos ? 1 : 0;
                                    if (mask === 0) {
                                        continue;
                                    }
                                    const pixel = dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                    dotProd += pixel * mask;
                                }
                            }
                        }
                        dx.set(dotProd, batch, dxDepth, dxRow, dxCol, channel);
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
const maxPool3DGradConfig = {
    kernelName: MaxPool3DGrad,
    backendName: 'cpu',
    kernelFunc: maxPool3DGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function maxPoolGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input, output } = inputs;
    const x = input;
    assertNotComplex([input, output], 'maxPoolGrad');
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    const convInfo = computePool2DInfo(x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
    const xValues = backend.data.get(x.dataId).values;
    const maxPosBuf = buffer(convInfo.outShape, x.dtype, maxPoolPositions(xValues, x.shape, x.dtype, convInfo).values);
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = buffer(x.shape, 'float32');
    const dyData = backend.data.get(dy.dataId).values;
    const dyBuf = buffer(dy.shape, 'float32', dyData);
    for (let b = 0; b < convInfo.batchSize; ++b) {
        for (let d = 0; d < convInfo.inChannels; ++d) {
            for (let dxR = 0; dxR < convInfo.inHeight; ++dxR) {
                for (let dxC = 0; dxC < convInfo.inWidth; ++dxC) {
                    // Shader code begins.
                    const dyRCorner = dxR - padTop;
                    const dyCCorner = dxC - padLeft;
                    let dotProd = 0;
                    for (let wR = 0; wR < effectiveFilterHeight; wR += dilationHeight) {
                        const dyR = (dyRCorner + wR) / strideHeight;
                        if (dyR < 0 || dyR >= convInfo.outHeight ||
                            Math.floor(dyR) !== dyR) {
                            continue;
                        }
                        for (let wC = 0; wC < effectiveFilterWidth; wC += dilationWidth) {
                            const dyC = (dyCCorner + wC) / strideWidth;
                            if (dyC < 0 || dyC >= convInfo.outWidth ||
                                Math.floor(dyC) !== dyC) {
                                continue;
                            }
                            const maxPos = effectiveFilterHeight * effectiveFilterWidth - 1 -
                                maxPosBuf.get(b, dyR, dyC, d);
                            const curPos = wR * effectiveFilterWidth + wC;
                            const mask = maxPos === curPos ? 1 : 0;
                            if (mask === 0) {
                                continue;
                            }
                            const pixel = dyBuf.get(b, dyR, dyC, d);
                            dotProd += pixel * mask;
                        }
                    }
                    dx.set(dotProd, b, dxR, dxC, d);
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
const maxPoolGradConfig = {
    kernelName: MaxPoolGrad,
    backendName: 'cpu',
    kernelFunc: maxPoolGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function maxPoolWithArgmaxImpl(xValues, xShape, dtype, includeBatchInIndex, convInfo) {
    const strides = computeStrides(xShape);
    const maxPools = pool(xValues, xShape, dtype, strides, convInfo, 'max');
    const maxPositions = maxPoolPositions(xValues, xShape, dtype, convInfo, true, includeBatchInIndex);
    return [maxPools.values, maxPositions.values];
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const maxPoolWithArgmaxConfig = {
    kernelName: MaxPoolWithArgmax,
    backendName: 'cpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { x } = inputs;
        const { filterSize, strides, pad, includeBatchInIndex } = attrs;
        const cpuBackend = backend;
        assertNotComplex(x, 'MaxPoolWithArgmax');
        const values = cpuBackend.data.get(x.dataId).values;
        const convInfo = computePool2DInfo(x.shape, filterSize, strides, [1, 1], pad);
        const [pooled, indexes] = maxPoolWithArgmaxImpl(values, x.shape, x.dtype, includeBatchInIndex, convInfo);
        const pooledDataId = cpuBackend.write(pooled, convInfo.outShape, x.dtype);
        const indexesDataId = cpuBackend.write(indexes, convInfo.outShape, x.dtype);
        return [
            { dataId: pooledDataId, shape: convInfo.outShape, dtype: x.dtype },
            { dataId: indexesDataId, shape: convInfo.outShape, dtype: 'int32' }
        ];
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function sum(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    assertNotComplex(x, 'sum');
    let $x;
    if (x.dtype === 'bool') {
        $x = cast({ inputs: { x }, backend, attrs: { dtype: 'int32' } });
    }
    else {
        $x = identity({ inputs: { x }, backend });
    }
    const xRank = $x.shape.length;
    const axes = parseAxisParam(axis, $x.shape);
    const permutation = getAxesPermutation(axes, xRank);
    let reductionAxes = axes;
    let permutedX = $x;
    if (permutation != null) {
        permutedX =
            transpose({ inputs: { x: $x }, backend, attrs: { perm: permutation } });
        reductionAxes = getInnerMostAxes(reductionAxes.length, xRank);
    }
    assertAxesAreInnerMostDims('sum', reductionAxes, permutedX.shape.length);
    const [outShape, reduceShape] = computeOutAndReduceShapes(permutedX.shape, reductionAxes);
    const resultDtype = upcastType(permutedX.dtype, 'int32');
    let result = zeros(backend, outShape, resultDtype);
    const reduceSize = sizeFromShape(reduceShape);
    const vals = backend.data.get(result.dataId).values;
    const aVals = backend.data.get(permutedX.dataId).values;
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let sum = 0;
        for (let j = 0; j < reduceSize; ++j) {
            sum += aVals[offset + j];
        }
        vals[i] = sum;
    }
    if (keepDims) {
        const newShape = expandShapeToKeepDim(result.shape, axes);
        const oldResult = result;
        result = reshape({ inputs: { x: result }, backend, attrs: { shape: newShape } });
        backend.disposeIntermediateTensorInfo(oldResult);
    }
    backend.disposeIntermediateTensorInfo($x);
    if (permutation != null) {
        backend.disposeIntermediateTensorInfo(permutedX);
    }
    return result;
}
const sumConfig = {
    kernelName: Sum,
    backendName: 'cpu',
    kernelFunc: sum
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function mean(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    const axes = parseAxisParam(axis, x.shape);
    const shapes = computeOutAndReduceShapes(x.shape, axes);
    const reduceShape = shapes[1];
    const reduceSize = sizeFromShape(reduceShape);
    const toDispose = [];
    const reduceSizeScalar = backend.makeTensorInfo([], 'float32', new Float32Array([reduceSize]));
    toDispose.push(reduceSizeScalar);
    const $x = cast({ inputs: { x }, backend, attrs: { dtype: 'float32' } });
    toDispose.push($x);
    const res = div({ inputs: { a: $x, b: reduceSizeScalar }, backend });
    toDispose.push(res);
    const result = sum({ inputs: { x: res }, backend, attrs: { axis, keepDims } });
    toDispose.forEach(t => backend.disposeIntermediateTensorInfo(t));
    return result;
}
const meanConfig = {
    kernelName: Mean,
    backendName: 'cpu',
    kernelFunc: mean
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function min(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    assertNotComplex(x, 'min');
    const origAxes = parseAxisParam(axis, x.shape);
    let axes = origAxes;
    const permutedAxes = getAxesPermutation(axes, x.shape.length);
    let $x = x;
    if (permutedAxes != null) {
        $x = transpose({ inputs: { x }, backend, attrs: { perm: permutedAxes } });
        axes = getInnerMostAxes(axes.length, x.shape.length);
    }
    assertAxesAreInnerMostDims('min', axes, $x.shape.length);
    const [outShape, reduceShape] = computeOutAndReduceShapes($x.shape, axes);
    const reduceSize = sizeFromShape(reduceShape);
    const vals = makeZerosTypedArray(sizeFromShape(outShape), $x.dtype);
    const aVals = backend.data.get($x.dataId).values;
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let min = aVals[offset];
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            if (value < min) {
                min = value;
            }
        }
        vals[i] = min;
    }
    if (permutedAxes != null) {
        backend.disposeIntermediateTensorInfo($x);
    }
    const result = backend.makeTensorInfo(outShape, $x.dtype, vals);
    if (keepDims) {
        const expandedShape = expandShapeToKeepDim(outShape, origAxes);
        const reshapedResult = reshape({ inputs: { x: result }, backend, attrs: { shape: expandedShape } });
        backend.disposeIntermediateTensorInfo(result);
        return reshapedResult;
    }
    return result;
}
const minConfig = {
    kernelName: Min,
    backendName: 'cpu',
    kernelFunc: min
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function mirrorPad(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { paddings, mode } = attrs;
    assertNotComplex(x, 'mirrorPad');
    const outShape = paddings.map((p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
    const start = paddings.map(p => p[0]);
    const end = paddings.map((p, i) => p[0] + x.shape[i]);
    const offset = mode === 'reflect' ? 0 : 1;
    const xVals = backend.data.get(x.dataId).values;
    const xRank = x.shape.length;
    const xStrides = computeStrides(x.shape);
    const resultSize = sizeFromShape(outShape);
    const resultRank = outShape.length;
    const resultStrides = computeStrides(outShape);
    const resVals = getTypedArrayFromDType(x.dtype, resultSize);
    for (let i = 0; i < resultSize; i++) {
        let coords = indexToLoc(i, resultRank, resultStrides);
        for (let i = 0; i < resultRank; i++) {
            if (coords[i] < start[i]) {
                coords[i] = start[i] * 2 - coords[i] - offset;
            }
            else if (coords[i] >= end[i]) {
                coords[i] = (end[i] - 1) * 2 - coords[i] + offset;
            }
        }
        coords = coords.map((c, i) => c - start[i]);
        const inIndex = locToIndex(coords, xRank, xStrides);
        resVals[i] = xVals[inIndex];
    }
    const outId = backend.write(resVals, outShape, x.dtype);
    return { dataId: outId, shape: outShape, dtype: x.dtype };
}
const mirrorPadConfig = {
    kernelName: MirrorPad,
    backendName: 'cpu',
    kernelFunc: mirrorPad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const modImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => {
    const rem = aValue % bValue;
    if ((aValue < 0 && bValue < 0) || (aValue >= 0 && bValue >= 0)) {
        return rem;
    }
    else {
        return (rem + bValue) % bValue;
    }
}));
const mod = binaryKernelFunc(Mod, modImpl);
const modConfig = {
    kernelName: Mod,
    backendName: 'cpu',
    kernelFunc: mod
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function softmax(args) {
    const { inputs, backend, attrs } = args;
    const { logits } = inputs;
    const { dim } = attrs;
    const logitsRank = logits.shape.length;
    let $dim = dim;
    if ($dim === -1) {
        $dim = logitsRank - 1;
    }
    if ($dim !== logitsRank - 1) {
        throw Error('Softmax along a non-last dimension is not yet supported. ' +
            `Logits was rank ${logitsRank} and dim was ${$dim}`);
    }
    const axes = parseAxisParam([$dim], logits.shape);
    const maxLogit = max({
        inputs: { x: logits },
        backend,
        attrs: { reductionIndices: axes, keepDims: false }
    });
    const expandedShape = expandShapeToKeepDim(maxLogit.shape, axes);
    const maxLogitReshaped = reshape({ inputs: { x: maxLogit }, backend, attrs: { shape: expandedShape } });
    const a = sub({ inputs: { a: logits, b: maxLogitReshaped }, backend });
    const b = exp({ inputs: { x: a }, backend });
    const sumExp = sum({ inputs: { x: b }, backend, attrs: { axis: axes, keepDims: false } });
    const sumReshaped = reshape({ inputs: { x: sumExp }, backend, attrs: { shape: expandedShape } });
    const result = div({ inputs: { a: b, b: sumReshaped }, backend });
    backend.disposeIntermediateTensorInfo(maxLogit);
    backend.disposeIntermediateTensorInfo(maxLogitReshaped);
    backend.disposeIntermediateTensorInfo(a);
    backend.disposeIntermediateTensorInfo(b);
    backend.disposeIntermediateTensorInfo(sumExp);
    backend.disposeIntermediateTensorInfo(sumReshaped);
    return result;
}
const softmaxConfig = {
    kernelName: Softmax,
    backendName: 'cpu',
    kernelFunc: softmax
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function multinomial(args) {
    const { inputs, backend, attrs } = args;
    const { logits } = inputs;
    const { numSamples, seed, normalized } = attrs;
    assertNotComplex(logits, 'multinomial');
    const probabilities = normalized ?
        logits :
        softmax({ inputs: { logits }, backend, attrs: { dim: -1 } });
    const batchSize = probabilities.shape[0];
    const numEvents = probabilities.shape[1];
    const probVals = backend.data.get(probabilities.dataId).values;
    const resShape = [batchSize, numSamples];
    const resVals = makeZerosTypedArray(sizeFromShape(resShape), 'int32');
    for (let b = 0; b < batchSize; ++b) {
        const offset = b * numEvents;
        // The cdf won't include the last event. It will be implicit if no other
        // event happened.
        const cdf = new Float32Array(numEvents - 1);
        cdf[0] = probVals[offset];
        for (let event = 1; event < cdf.length; ++event) {
            cdf[event] = cdf[event - 1] + probVals[offset + event];
        }
        const random = seedrandom.alea(seed.toString());
        const outOffset = b * numSamples;
        for (let sampleId = 0; sampleId < numSamples; ++sampleId) {
            const r = random();
            // Assume last event happened by default.
            resVals[outOffset + sampleId] = cdf.length;
            for (let event = 0; event < cdf.length; event++) {
                if (r < cdf[event]) {
                    resVals[outOffset + sampleId] = event;
                    break;
                }
            }
        }
    }
    if (!normalized) {
        backend.disposeIntermediateTensorInfo(probabilities);
    }
    return backend.makeTensorInfo(resShape, 'int32', resVals);
}
const multinomialConfig = {
    kernelName: Multinomial,
    backendName: 'cpu',
    kernelFunc: multinomial
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const nonMaxSuppressionV3Impl = nonMaxSuppressionV3Impl$1;
function nonMaxSuppressionV3(args) {
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold } = attrs;
    assertNotComplex(boxes, 'NonMaxSuppression');
    const boxesVals = backend.data.get(boxes.dataId).values;
    const scoresVals = backend.data.get(scores.dataId).values;
    const { selectedIndices } = nonMaxSuppressionV3Impl(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold);
    return backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices));
}
const nonMaxSuppressionV3Config = {
    kernelName: NonMaxSuppressionV3,
    backendName: 'cpu',
    kernelFunc: nonMaxSuppressionV3
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const nonMaxSuppressionV4Impl = nonMaxSuppressionV4Impl$1;
function nonMaxSuppressionV4(args) {
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold, padToMaxOutputSize } = attrs;
    assertNotComplex(boxes, 'NonMaxSuppressionPadded');
    const boxesVals = backend.data.get(boxes.dataId).values;
    const scoresVals = backend.data.get(scores.dataId).values;
    const { selectedIndices, validOutputs } = nonMaxSuppressionV4Impl(boxesVals, scoresVals, maxOutputSize, iouThreshold, scoreThreshold, padToMaxOutputSize);
    return [
        backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices)),
        backend.makeTensorInfo([], 'int32', new Int32Array([validOutputs]))
    ];
}
const nonMaxSuppressionV4Config = {
    kernelName: NonMaxSuppressionV4,
    backendName: 'cpu',
    kernelFunc: nonMaxSuppressionV4
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const nonMaxSuppressionV5Impl = nonMaxSuppressionV5Impl$1;
function nonMaxSuppressionV5(args) {
    const { inputs, backend, attrs } = args;
    const { boxes, scores } = inputs;
    const { maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma } = attrs;
    assertNotComplex(boxes, 'NonMaxSuppressionWithScore');
    const boxesVals = backend.data.get(boxes.dataId).values;
    const scoresVals = backend.data.get(scores.dataId).values;
    const maxOutputSizeVal = maxOutputSize;
    const iouThresholdVal = iouThreshold;
    const scoreThresholdVal = scoreThreshold;
    const softNmsSigmaVal = softNmsSigma;
    const { selectedIndices, selectedScores } = nonMaxSuppressionV5Impl(boxesVals, scoresVals, maxOutputSizeVal, iouThresholdVal, scoreThresholdVal, softNmsSigmaVal);
    return [
        backend.makeTensorInfo([selectedIndices.length], 'int32', new Int32Array(selectedIndices)),
        backend.makeTensorInfo([selectedScores.length], 'float32', new Float32Array(selectedScores))
    ];
}
const nonMaxSuppressionV5Config = {
    kernelName: NonMaxSuppressionV5,
    backendName: 'cpu',
    kernelFunc: nonMaxSuppressionV5
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function oneHot(args) {
    const { inputs, backend, attrs } = args;
    const { indices } = inputs;
    const { depth, onValue, offValue } = attrs;
    assertNotComplex(indices, 'oneHot');
    const indicesSize = sizeFromShape(indices.shape);
    const res = new Float32Array(indicesSize * depth);
    res.fill(offValue);
    const indicesVal = backend.data.get(indices.dataId).values;
    for (let event = 0; event < indicesSize; ++event) {
        if (indicesVal[event] >= 0 && indicesVal[event] < depth) {
            res[event * depth + indicesVal[event]] = onValue;
        }
    }
    return backend.makeTensorInfo([...indices.shape, depth], 'int32', res);
}
const oneHotConfig = {
    kernelName: OneHot,
    backendName: 'cpu',
    kernelFunc: oneHot
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function zerosLike(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (x.dtype === 'string') {
        throw new Error('zerosLike is not supported for string tensors');
    }
    else if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const r = zerosLike({ inputs: { x: realPart }, backend });
        const imagPart = imag({ inputs: { input: x }, backend });
        const i = zerosLike({ inputs: { x: imagPart }, backend });
        const result = complex({ inputs: { real: r, imag: i }, backend });
        backend.disposeIntermediateTensorInfo(realPart);
        backend.disposeIntermediateTensorInfo(r);
        backend.disposeIntermediateTensorInfo(imagPart);
        backend.disposeIntermediateTensorInfo(i);
        return result;
    }
    else {
        return fill({ backend, attrs: { shape: x.shape, value: 0, dtype: x.dtype } });
    }
}
const zerosLikeConfig = {
    kernelName: ZerosLike,
    backendName: 'cpu',
    kernelFunc: zerosLike
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function onesLike(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    if (x.dtype === 'string') {
        throw new Error('onesLike is not supported for string tensors');
    }
    else if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const r = onesLike({ inputs: { x: realPart }, backend });
        const imagPart = imag({ inputs: { input: x }, backend });
        const i = zerosLike({ inputs: { x: imagPart }, backend });
        const result = complex({ inputs: { real: r, imag: i }, backend });
        backend.disposeIntermediateTensorInfo(realPart);
        backend.disposeIntermediateTensorInfo(r);
        backend.disposeIntermediateTensorInfo(imagPart);
        backend.disposeIntermediateTensorInfo(i);
        return result;
    }
    else {
        return fill({ backend, attrs: { shape: x.shape, value: 1, dtype: x.dtype } });
    }
}
const onesLikeConfig = {
    kernelName: OnesLike,
    backendName: 'cpu',
    kernelFunc: onesLike
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function pack(args) {
    const { inputs, backend, attrs } = args;
    const { axis } = attrs;
    if (inputs.length === 1) {
        return expandDims({ inputs: { input: inputs[0] }, backend, attrs: { dim: axis } });
    }
    const shape = inputs[0].shape;
    const dtype = inputs[0].dtype;
    inputs.forEach(t => {
        assertShapesMatch(shape, t.shape, 'All tensors passed to stack must have matching shapes');
        assert(dtype === t.dtype, () => 'All tensors passed to stack must have matching dtypes');
    });
    const intermediateTensorInfos = [];
    const expandedTensors = inputs.map(t => {
        const expandedT = expandDims({ inputs: { input: t }, backend, attrs: { dim: axis } });
        intermediateTensorInfos.push(expandedT);
        return expandedT;
    });
    const result = concat({ inputs: expandedTensors, backend, attrs: { axis } });
    intermediateTensorInfos.forEach(t => backend.disposeIntermediateTensorInfo(t));
    return result;
}
const packConfig = {
    kernelName: Pack,
    backendName: 'cpu',
    kernelFunc: pack
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function padV2(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { paddings, constantValue } = attrs;
    assertNotComplex(x, 'pad');
    const outShape = paddings.map((p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
    const start = paddings.map(p => p[0]);
    const xVals = backend.data.get(x.dataId).values;
    const xSize = sizeFromShape(x.shape);
    const xRank = x.shape.length;
    const xStrides = computeStrides(x.shape);
    const resultSize = sizeFromShape(outShape);
    const resultRank = outShape.length;
    const resultStrides = computeStrides(outShape);
    const resVals = getTypedArrayFromDType(x.dtype, resultSize);
    if (constantValue !== 0) {
        resVals.fill(constantValue);
    }
    for (let i = 0; i < xSize; i++) {
        const coords = indexToLoc(i, xRank, xStrides);
        const outCoords = coords.map((c, i) => c + start[i]);
        const outIndex = locToIndex(outCoords, resultRank, resultStrides);
        resVals[outIndex] = xVals[i];
    }
    const outId = backend.write(resVals, outShape, x.dtype);
    return { dataId: outId, shape: outShape, dtype: x.dtype };
}
const padV2Config = {
    kernelName: PadV2,
    backendName: 'cpu',
    kernelFunc: padV2
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const powImpl = createSimpleBinaryKernelImpl((a, b) => Math.pow(a, b));
const pow = binaryKernelFunc(Pow, powImpl);
const powConfig = {
    kernelName: Pow,
    backendName: 'cpu',
    kernelFunc: pow
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function range(args) {
    const { backend, attrs } = args;
    const { start, stop, dtype, step } = attrs;
    const values = rangeImpl(start, stop, step, dtype);
    return backend.makeTensorInfo([values.length], dtype, values);
}
const rangeConfig = {
    kernelName: Range,
    backendName: 'cpu',
    kernelFunc: range
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const reciprocal = unaryKernelFunc(Reciprocal, (xi) => 1 / xi);
const reciprocalConfig = {
    kernelName: Reciprocal,
    backendName: 'cpu',
    kernelFunc: reciprocal,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function resizeBilinear(args) {
    const { inputs, backend, attrs } = args;
    const { images } = inputs;
    const { alignCorners, halfPixelCenters, size } = attrs;
    assertNotComplex(images, 'resizeBilinear');
    const imagesStrides = computeStrides(images.shape);
    const [newHeight, newWidth] = size;
    const [batch, oldHeight, oldWidth, numChannels] = images.shape;
    const xValues = backend.data.get(images.dataId).values;
    const result = new Float32Array(sizeFromShape([batch, newHeight, newWidth, numChannels]));
    const effectiveInputSize = [
        (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
        (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
    ];
    const effectiveOutputSize = [
        (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
        (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
    ];
    let outputIdx = 0;
    const effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
    const effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
    for (let b = 0; b < batch; b++) {
        for (let r = 0; r < newHeight; r++) {
            let sourceFracRow;
            if (halfPixelCenters) {
                sourceFracRow = effectiveRowSizeRatio * (r + 0.5) - 0.5;
            }
            else {
                sourceFracRow = effectiveRowSizeRatio * r;
            }
            const sourceRowFloor = Math.max(0, Math.floor(sourceFracRow));
            const rowFrac = sourceFracRow - sourceRowFloor;
            const sourceRowCeil = Math.min(oldHeight - 1, Math.ceil(sourceFracRow));
            const topRowOffset = b * imagesStrides[0] + sourceRowFloor * imagesStrides[1];
            const botRowOffset = b * imagesStrides[0] + sourceRowCeil * imagesStrides[1];
            for (let c = 0; c < newWidth; c++) {
                let sourceFracCol;
                if (halfPixelCenters) {
                    sourceFracCol = effectiveColSizeRatio * (c + 0.5) - 0.5;
                }
                else {
                    sourceFracCol = effectiveColSizeRatio * c;
                }
                const sourceColFloor = Math.max(0, Math.floor(sourceFracCol));
                const colFrac = sourceFracCol - sourceColFloor;
                const sourceColCeil = Math.min(oldWidth - 1, Math.ceil(sourceFracCol));
                const topLeftOffest = topRowOffset + sourceColFloor * imagesStrides[2];
                const botLeftOffset = botRowOffset + sourceColFloor * imagesStrides[2];
                const topRightOffset = topRowOffset + sourceColCeil * imagesStrides[2];
                const botRightOffest = botRowOffset + sourceColCeil * imagesStrides[2];
                for (let d = 0; d < numChannels; d++) {
                    // Begin shader.
                    // Compute the fractional index of the source.
                    const topLeft = xValues[topLeftOffest + d];
                    const bottomLeft = xValues[botLeftOffset + d];
                    const topRight = xValues[topRightOffset + d];
                    const bottomRight = xValues[botRightOffest + d];
                    const top = topLeft + (topRight - topLeft) * colFrac;
                    const bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                    const newValue = top + (bottom - top) * rowFrac;
                    result[outputIdx++] = newValue;
                }
            }
        }
    }
    return backend.makeTensorInfo([batch, newHeight, newWidth, numChannels], 'float32', result);
}
const resizeBilinearConfig = {
    kernelName: ResizeBilinear,
    backendName: 'cpu',
    kernelFunc: resizeBilinear
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function resizeBilinearGrad(args) {
    const { inputs, backend, attrs } = args;
    const { images, dy } = inputs;
    const { alignCorners } = attrs;
    assertNotComplex([dy, images], 'resizeBilinearGrad');
    const imagesStrides = computeStrides(images.shape);
    const [batch, xHeight, xWidth, depth] = images.shape;
    const [, yHeight, yWidth] = dy.shape;
    const output = new Float32Array(batch * xHeight * xWidth * depth);
    // In the backwards pass, we want to find the pixels that were generated
    // for each pixel in the input image the forward pass and add the
    // corresponding coefficient from dy to the gradient (with some
    // interpolation).
    const effectiveXSize = [
        (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
        (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
    ];
    const effectiveYSize = [
        (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
        (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
    ];
    const heightScale = effectiveXSize[0] / effectiveYSize[0];
    const widthScale = effectiveXSize[1] / effectiveYSize[1];
    // Reference implementation
    // tslint:disable-next-line:max-line-length
    // https://github.com/tensorflow/tensorflow/blob/3039375c86a5bbc9610c7725dcaa95d635f87ba2/tensorflow/core/kernels/resize_bilinear_op.cc#L275
    const dyValues = backend.data.get(dy.dataId).values;
    let offset = 0;
    for (let b = 0; b < batch; b++) {
        const bOffset = b * imagesStrides[0];
        for (let r = 0; r < yHeight; r++) {
            const dxR = r * heightScale;
            const topDxRIndex = Math.floor(dxR);
            const bottomDxRIndex = Math.min(Math.ceil(dxR), xHeight - 1);
            const topDxROffset = bOffset + topDxRIndex * imagesStrides[1];
            const bottomDxROffset = bOffset + bottomDxRIndex * imagesStrides[1];
            const dxRLerp = dxR - topDxRIndex;
            const inverseDxRLerp = 1.0 - dxRLerp;
            for (let c = 0; c < yWidth; c++) {
                const dxC = c * widthScale;
                const leftDxCIndex = Math.floor(dxC);
                const rightDxCIndex = Math.min(Math.ceil(dxC), xWidth - 1);
                const dxCLerp = dxC - leftDxCIndex;
                const inverseDxCLerp = 1.0 - dxCLerp;
                const topLeftRCOffset = topDxROffset + leftDxCIndex * imagesStrides[2];
                const topRightRCOffset = topDxROffset + rightDxCIndex * imagesStrides[2];
                const bottomLeftRCOffset = bottomDxROffset + leftDxCIndex * imagesStrides[2];
                const bottomRightRCOffset = bottomDxROffset + rightDxCIndex * imagesStrides[2];
                const inverseDxRLerpTimesInverseDxCLerp = inverseDxRLerp * inverseDxCLerp;
                const inverseDxRLerpTimesDxCLerp = inverseDxRLerp * dxCLerp;
                const dxRLerpTimesInverseDxCLerp = dxRLerp * inverseDxCLerp;
                const dxRLerpTimesDxCLerp = dxRLerp * dxCLerp;
                for (let d = 0; d < depth; d++) {
                    const dyVal = dyValues[offset++];
                    output[topLeftRCOffset + d] +=
                        dyVal * inverseDxRLerpTimesInverseDxCLerp;
                    output[topRightRCOffset + d] += dyVal * inverseDxRLerpTimesDxCLerp;
                    output[bottomLeftRCOffset + d] += dyVal * dxRLerpTimesInverseDxCLerp;
                    output[bottomRightRCOffset + d] += dyVal * dxRLerpTimesDxCLerp;
                }
            }
        }
    }
    return backend.makeTensorInfo([batch, xWidth, xHeight, depth], 'float32', output);
}
const resizeBilinearGradConfig = {
    kernelName: ResizeBilinearGrad,
    backendName: 'cpu',
    kernelFunc: resizeBilinearGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function resizeNearestNeighbor(args) {
    const { inputs, backend, attrs } = args;
    const { images } = inputs;
    const { alignCorners, halfPixelCenters, size } = attrs;
    assertNotComplex(images, 'resizeNearestNeighbor');
    const imagesStrides = computeStrides(images.shape);
    const [newHeight, newWidth] = size;
    const [batch, oldHeight, oldWidth, numChannels] = images.shape;
    const xValues = backend.data.get(images.dataId).values;
    const output = new Float32Array(batch * newHeight * newWidth * numChannels);
    const effectiveInputSize = [
        (alignCorners && newHeight > 1) ? oldHeight - 1 : oldHeight,
        (alignCorners && newWidth > 1) ? oldWidth - 1 : oldWidth
    ];
    const effectiveOutputSize = [
        (alignCorners && newHeight > 1) ? newHeight - 1 : newHeight,
        (alignCorners && newWidth > 1) ? newWidth - 1 : newWidth
    ];
    const effectiveRowSizeRatio = effectiveInputSize[0] / effectiveOutputSize[0];
    const effectiveColSizeRatio = effectiveInputSize[1] / effectiveOutputSize[1];
    let outputOffset = 0;
    for (let b = 0; b < batch; b++) {
        const batchOffset = b * imagesStrides[0];
        for (let r = 0; r < newHeight; r++) {
            const sourceFracRow = halfPixelCenters ?
                effectiveRowSizeRatio * (r + 0.5) :
                effectiveRowSizeRatio * r;
            let sourceNearestRow = Math.min(oldHeight - 1, alignCorners ? Math.round(sourceFracRow) : Math.floor(sourceFracRow));
            if (halfPixelCenters) {
                sourceNearestRow = Math.max(0, sourceNearestRow);
            }
            const rowOffset = batchOffset + sourceNearestRow * imagesStrides[1];
            for (let c = 0; c < newWidth; c++) {
                const sourceFracCol = halfPixelCenters ?
                    effectiveColSizeRatio * (c + 0.5) :
                    effectiveColSizeRatio * c;
                let sourceNearestCol = Math.min(oldWidth - 1, alignCorners ? Math.round(sourceFracCol) :
                    Math.floor(sourceFracCol));
                if (halfPixelCenters) {
                    sourceNearestCol = Math.max(0, sourceNearestCol);
                }
                const colOffset = rowOffset + sourceNearestCol * imagesStrides[2];
                for (let d = 0; d < numChannels; d++) {
                    // Begin shader.
                    // Compute the fractional index of the source.
                    const newVal = xValues[colOffset + d];
                    output[outputOffset++] = newVal;
                }
            }
        }
    }
    return backend.makeTensorInfo([batch, newHeight, newWidth, numChannels], images.dtype, output);
}
const resizeNearestNeighborConfig = {
    kernelName: ResizeNearestNeighbor,
    backendName: 'cpu',
    kernelFunc: resizeNearestNeighbor
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function resizeNearestNeighborGrad(args) {
    const { inputs, backend, attrs } = args;
    const { images, dy } = inputs;
    const { alignCorners } = attrs;
    assertNotComplex([dy, images], 'resizeNearestNeighborGrad');
    const imagesStrides = computeStrides(images.shape);
    const dyStrides = computeStrides(dy.shape);
    const [batch, xHeight, xWidth, depth] = images.shape;
    const [, yHeight, yWidth] = dy.shape;
    const output = new Float32Array(batch * xHeight * xWidth * depth);
    const dyValues = backend.data.get(dy.dataId).values;
    // In the backwards pass, we want to find the pixels that were generated
    // for each pixel in the input image the forward pass
    const effectiveXSize = [
        (alignCorners && yHeight > 1) ? xHeight - 1 : xHeight,
        (alignCorners && yWidth > 1) ? xWidth - 1 : xWidth
    ];
    const effectiveYSize = [
        (alignCorners && yHeight > 1) ? yHeight - 1 : yHeight,
        (alignCorners && yWidth > 1) ? yWidth - 1 : yWidth
    ];
    const heightScale = effectiveXSize[0] / effectiveYSize[0];
    const widthScale = effectiveXSize[1] / effectiveYSize[1];
    const invHeightScale = 1 / heightScale;
    const invWidthScale = 1 / widthScale;
    // This defines the size of the window of values around a particular
    // index in dy that we want to search for contributions to dx.
    const winHeight = (Math.ceil(invHeightScale) * 2) + 2;
    const winWidth = (Math.ceil(invWidthScale) * 2) + 2;
    // Loop over the output space.
    for (let b = 0; b < batch; b++) {
        const batchOffset = b * imagesStrides[0];
        for (let r = 0; r < xHeight; r++) {
            const rowOffset = batchOffset + r * imagesStrides[1];
            // Compute bounds for where in dy we will look
            const startRLerp = Math.floor(r * invHeightScale);
            const startDyR = Math.floor(startRLerp - (winHeight / 2));
            for (let c = 0; c < xWidth; c++) {
                const colOffset = rowOffset + c * imagesStrides[2];
                // Compute bounds for where in dy we will look
                const startCLerp = Math.floor(c * invWidthScale);
                const startDyC = Math.floor(startCLerp - (winWidth / 2));
                for (let d = 0; d < depth; d++) {
                    let accum = 0;
                    // loop over dy
                    for (let dyRIndex = 0; dyRIndex < winHeight; dyRIndex++) {
                        const dyR = dyRIndex + startDyR;
                        // Guard against the window exceeding the bounds of dy
                        if (dyR < 0 || dyR >= yHeight) {
                            continue;
                        }
                        const dyROffset = batchOffset + dyR * dyStrides[1];
                        const sourceFracRow = dyR * heightScale;
                        const sourceNearestRow = Math.min(xHeight - 1, alignCorners ? Math.round(sourceFracRow) :
                            Math.floor(sourceFracRow));
                        if (r !== sourceNearestRow) {
                            continue;
                        }
                        for (let dyCIndex = 0; dyCIndex < winWidth; dyCIndex++) {
                            const dyC = dyCIndex + startDyC;
                            // Guard against the window exceeding the bounds of dy
                            if (dyC < 0 || dyC >= yWidth) {
                                continue;
                            }
                            const dyCOffset = dyROffset + dyC * dyStrides[2];
                            const sourceFracCol = dyC * widthScale;
                            const sourceNearestCol = Math.min(xWidth - 1, alignCorners ? Math.round(sourceFracCol) :
                                Math.floor(sourceFracCol));
                            if (c === sourceNearestCol) {
                                accum += dyValues[dyCOffset + d];
                            }
                        }
                    }
                    output[colOffset + d] = accum;
                }
            }
        }
    }
    return backend.makeTensorInfo(images.shape, images.dtype, output);
}
const resizeNearestNeighborGradConfig = {
    kernelName: ResizeNearestNeighborGrad,
    backendName: 'cpu',
    kernelFunc: resizeNearestNeighborGrad
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function reverse(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { dims } = attrs;
    assertNotComplex(x, 'reverse');
    const xRank = x.shape.length;
    const $dims = parseAxisParam(dims, x.shape);
    if (xRank === 0) {
        return identity({ inputs: { x }, backend });
    }
    const outBuf = new TensorBuffer(x.shape, x.dtype);
    const xBuf = backend.bufferSync(x);
    for (let i = 0; i < outBuf.size; i++) {
        const outLoc = outBuf.indexToLoc(i);
        const inLoc = outLoc.slice();
        $dims.forEach(d => inLoc[d] = x.shape[d] - 1 - inLoc[d]);
        outBuf.set(xBuf.get(...inLoc), ...outLoc);
    }
    return backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
}
const reverseConfig = {
    kernelName: Reverse,
    backendName: 'cpu',
    kernelFunc: reverse
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const rotateWithOffsetConfig = {
    kernelName: RotateWithOffset,
    backendName: 'cpu',
    kernelFunc: ({ inputs, attrs, backend }) => {
        const { image } = inputs;
        const { radians, fillValue, center } = attrs;
        const cpuBackend = backend;
        const output = getTypedArrayFromDType(image.dtype, sizeFromShape(image.shape));
        const [batch, imageHeight, imageWidth, numChannels] = image.shape;
        const [centerX, centerY] = getImageCenter(center, imageHeight, imageWidth);
        const fullOpacityValue = 255;
        const sinFactor = Math.sin(radians);
        const cosFactor = Math.cos(radians);
        const imageVals = cpuBackend.data.get(image.dataId).values;
        for (let batchIdx = 0; batchIdx < batch; batchIdx++) {
            const batchOffset = batchIdx * imageWidth * imageHeight * numChannels;
            for (let row = 0; row < imageHeight; row++) {
                const rowOffset = row * (imageWidth * numChannels);
                for (let col = 0; col < imageWidth; col++) {
                    const colOffset = col * numChannels;
                    for (let channel = 0; channel < numChannels; channel++) {
                        const coords = [batch, row, col, channel];
                        const x = coords[2];
                        const y = coords[1];
                        // coordX/coordY are the result of rotating and translating x/y.
                        let coordX = (x - centerX) * cosFactor - (y - centerY) * sinFactor;
                        let coordY = (x - centerX) * sinFactor + (y - centerY) * cosFactor;
                        coordX = Math.round(coordX + centerX);
                        coordY = Math.round(coordY + centerY);
                        let outputValue = fillValue;
                        if (typeof fillValue !== 'number') {
                            if (channel === 3) {
                                outputValue = fullOpacityValue;
                            }
                            else {
                                outputValue = fillValue[channel];
                            }
                        }
                        // If the coordinate position falls within the image boundaries...
                        if (coordX >= 0 && coordX < imageWidth && coordY >= 0 &&
                            coordY < imageHeight) {
                            // set the output to the image value at the coordinate position.
                            const rotatedRowOffset = coordY * (imageWidth * numChannels);
                            const rotatedColOffset = coordX * numChannels;
                            const imageIdx = batchOffset + rotatedRowOffset + rotatedColOffset + channel;
                            outputValue = imageVals[imageIdx];
                        }
                        const outIdx = batchOffset + rowOffset + colOffset + channel;
                        output[outIdx] = outputValue;
                    }
                }
            }
        }
        const dataId = cpuBackend.write(output, image.shape, image.dtype);
        return { dataId, shape: image.shape, dtype: image.dtype };
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const round = unaryKernelFunc(Round, (xi) => {
    // The algorithm is based on banker's rounding.
    const base = Math.floor(xi);
    if (xi - base < 0.5) {
        return Math.floor(xi);
    }
    else if (xi - base > 0.5) {
        return Math.ceil(xi);
    }
    else {
        if (base % 2.0 === 0.0) {
            return base;
        }
        else {
            return base + 1.0;
        }
    }
});
const roundConfig = {
    kernelName: Round,
    backendName: 'cpu',
    kernelFunc: round,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function scatterImpl(indices, updates, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, defaultValue, sumDupeIndices) {
    const flattenShape = [outputSize / sliceSize, sliceSize];
    const indicesData = indices.values;
    const updatesData = updates.values;
    if (outputSize === 0) {
        return buffer(shape, updates.dtype);
    }
    const outBuf = buffer(flattenShape, updates.dtype);
    outBuf.values.fill(defaultValue);
    for (let i = 0; i < numUpdates; i++) {
        const index = [];
        let flattenIndex = 0;
        for (let j = 0; j < sliceRank; j++) {
            const dim = indicesData[i * sliceRank + j];
            index.push(dim);
            flattenIndex += dim * strides[j];
        }
        if (flattenIndex < 0 || flattenIndex >= outputSize / sliceSize) {
            throw new Error(`Invalid indices: ${index} does not index into ${shape}`);
        }
        for (let k = 0; k < sliceSize; k++) {
            if (sumDupeIndices) {
                outBuf.values[flattenIndex * sliceSize + k] +=
                    updatesData[i * sliceSize + k];
            }
            else {
                outBuf.values[flattenIndex * sliceSize + k] = updates.rank === 0 ?
                    updatesData[0] :
                    updatesData[i * sliceSize + k];
            }
        }
    }
    return outBuf;
}

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function scatterNd(args) {
    const { inputs, backend, attrs } = args;
    const { indices, updates } = inputs;
    const { shape } = attrs;
    const { sliceRank, numUpdates, sliceSize, strides, outputSize } = calculateShapes(updates, indices, shape);
    const sumDupeIndices = true;
    const indicesBuf = backend.bufferSync(indices);
    const updatesBuf = backend.bufferSync(updates);
    const outBuf = scatterImpl(indicesBuf, updatesBuf, shape, outputSize, sliceSize, numUpdates, sliceRank, strides, 0 /* defaultValue */, sumDupeIndices);
    return backend.makeTensorInfo(shape, outBuf.dtype, outBuf.values);
}
const scatterNdConfig = {
    kernelName: ScatterNd,
    backendName: 'cpu',
    kernelFunc: scatterNd
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function select(args) {
    const { inputs, backend } = args;
    const { condition, t, e } = inputs;
    assertNotComplex([condition, t, e], 'select');
    const conditionRank = condition.shape.length;
    const values = backend.data.get(condition.dataId).values;
    const tValues = backend.data.get(t.dataId).values;
    const eValues = backend.data.get(e.dataId).values;
    const resultDtype = upcastType(t.dtype, e.dtype);
    const newValues = makeZerosTypedArray(sizeFromShape(t.shape), resultDtype);
    let index = 0;
    const offset = conditionRank === 0 || conditionRank > 1 || t.shape.length === 1 ?
        1 :
        sizeFromShape(t.shape.slice(1));
    for (let i = 0; i < values.length; i++) {
        for (let j = 0; j < offset; j++) {
            if (values[i] === 1) {
                newValues[index++] = tValues[i];
            }
            else {
                newValues[index++] = eValues[i];
            }
        }
    }
    return backend.makeTensorInfo(t.shape, resultDtype, newValues);
}
const selectConfig = {
    kernelName: Select,
    backendName: 'cpu',
    kernelFunc: select
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const scaleAlpha = SELU_SCALEALPHA;
const scale = SELU_SCALE;
const selu = unaryKernelFunc(Selu, (xi) => {
    if (xi >= 0) {
        return scale * xi;
    }
    else {
        return scaleAlpha * (Math.exp(xi) - 1);
    }
});
const seluConfig = {
    kernelName: Selu,
    backendName: 'cpu',
    kernelFunc: selu,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sigmoid = unaryKernelFunc(Sigmoid, (xi) => 1 / (1 + Math.exp(-xi)));
const sigmoidConfig = {
    kernelName: Sigmoid,
    backendName: 'cpu',
    kernelFunc: sigmoid,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sign = unaryKernelFunc(Sign, (xi) => {
    if (xi < 0) {
        return -1;
    }
    else if (xi > 0) {
        return 1;
    }
    else {
        return 0;
    }
});
const signConfig = {
    kernelName: Sign,
    backendName: 'cpu',
    kernelFunc: sign,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sin = unaryKernelFunc(Sin, (xi) => Math.sin(xi));
const sinConfig = {
    kernelName: Sin,
    backendName: 'cpu',
    kernelFunc: sin,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sinh = unaryKernelFunc(Sinh, (xi) => Math.sinh(xi));
const sinhConfig = {
    kernelName: Sinh,
    backendName: 'cpu',
    kernelFunc: sinh,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// mirrors the implementation of tf.nn.softplus: https://goo.gl/vkcvwX
// epsilon is the difference between 1.0 and the next representable float.
// For a single precision 32 bit float this should be 2^-23, see:
// https://math.byu.edu/~schow/work/IEEEFloatingPoint.htm
const epsilon = 1.1920928955078125e-7;
const threshold = Math.log(epsilon) + 2.0;
const softplus = unaryKernelFunc(Softplus, (xi) => {
    // Value above which exp(x) may overflow, but softplus(x) == x
    // is within machine epsilon.
    const tooLarge = xi > -threshold;
    // Value below which exp(x) may underflow, but softplus(x) == exp(x)
    // is within machine epsilon.
    const tooSmall = xi < threshold;
    const expX = Math.exp(xi);
    let result;
    if (tooSmall) {
        result = expX;
    }
    else if (tooLarge) {
        result = xi;
    }
    else {
        result = Math.log(1.0 + expX);
    }
    return result;
});
const softplusConfig = {
    kernelName: Softplus,
    backendName: 'cpu',
    kernelFunc: softplus,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function spaceToBatchND(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { blockShape, paddings } = attrs;
    assertNotComplex([x], 'spaceToBatchND');
    const prod = sizeFromShape(blockShape);
    const completePaddings = [[0, 0]];
    completePaddings.push(...paddings);
    for (let i = 1 + blockShape.length; i < x.shape.length; ++i) {
        completePaddings.push([0, 0]);
    }
    const paddedX = padV2Config.kernelFunc({
        inputs: { x },
        backend,
        attrs: { paddings: completePaddings, constantValue: 0 }
    });
    const reshapedPaddedShape = getReshaped(paddedX.shape, blockShape, prod, false);
    const permutedReshapedPaddedPermutation = getPermuted(reshapedPaddedShape.length, blockShape.length, false);
    const flattenShape = getReshapedPermuted(paddedX.shape, blockShape, prod, false);
    const reshapeInputs = { x: paddedX };
    const reshapeAttrs = { shape: reshapedPaddedShape };
    const paddedXReshaped = reshape({ inputs: reshapeInputs, backend, attrs: reshapeAttrs });
    const transposeInputs = { x: paddedXReshaped };
    const transposeAttrs = { perm: permutedReshapedPaddedPermutation };
    const paddedXT = transpose({ inputs: transposeInputs, backend, attrs: transposeAttrs });
    const resultReshapeInputs = { x: paddedXT };
    const resultReshapeAttrs = { shape: flattenShape };
    const result = reshape({ inputs: resultReshapeInputs, backend, attrs: resultReshapeAttrs });
    backend.disposeIntermediateTensorInfo(paddedX);
    backend.disposeIntermediateTensorInfo(paddedXReshaped);
    backend.disposeIntermediateTensorInfo(paddedXT);
    return result;
}
const spaceToBatchNDConfig = {
    kernelName: SpaceToBatchND,
    backendName: 'cpu',
    kernelFunc: spaceToBatchND
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function sparseToDense(args) {
    const { inputs, backend, attrs } = args;
    const { sparseIndices, sparseValues, defaultValue } = inputs;
    const { outputShape } = attrs;
    const { sliceRank, numUpdates, sliceSize, strides, outputSize } = calculateShapes(sparseValues, sparseIndices, outputShape);
    const sumDupeIndices = false;
    const indicesBuf = backend.bufferSync(sparseIndices);
    const updatesBuf = backend.bufferSync(sparseValues);
    const $defaultValue = backend.data.get(defaultValue.dataId).values[0];
    const outBuf = scatterImpl(indicesBuf, updatesBuf, outputShape, outputSize, sliceSize, numUpdates, sliceRank, strides, $defaultValue, sumDupeIndices);
    return backend.makeTensorInfo(outputShape, outBuf.dtype, outBuf.values);
}
const sparseToDenseConfig = {
    kernelName: SparseToDense,
    backendName: 'cpu',
    kernelFunc: sparseToDense
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function splitV(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { numOrSizeSplits, axis } = attrs;
    const $axis = parseAxisParam(axis, x.shape)[0];
    const splitSizes = prepareSplitSize(x, numOrSizeSplits, $axis);
    const begin = new Array(x.shape.length).fill(0);
    const size = x.shape.slice();
    return splitSizes.map(s => {
        const sliceSize = [...size];
        sliceSize[$axis] = s;
        const sliceT = slice({ inputs: { x }, backend, attrs: { begin, size: sliceSize } });
        begin[$axis] += s;
        return sliceT;
    });
}
const splitVConfig = {
    kernelName: SplitV,
    backendName: 'cpu',
    kernelFunc: splitV
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const sqrt = unaryKernelFunc(Sqrt, (xi) => Math.sqrt(xi));
const sqrtConfig = {
    kernelName: Sqrt,
    backendName: 'cpu',
    kernelFunc: sqrt,
};

/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const squareConfig = {
    kernelName: Square,
    backendName: 'cpu',
    kernelFunc: ({ inputs, backend }) => {
        const { x } = inputs;
        const cpuBackend = backend;
        assertNotComplex(x, 'square');
        const values = cpuBackend.data.get(x.dataId).values;
        const newValues = new Float32Array(values.length);
        for (let i = 0; i < values.length; ++i) {
            const value = values[i];
            newValues[i] = value * value;
        }
        const dataId = cpuBackend.write(newValues, x.shape, x.dtype);
        return { dataId, shape: x.shape, dtype: x.dtype };
    }
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const step = unaryKernelFunc(Step, (xi, attrs) => {
    const stepAttrs = attrs;
    if (isNaN(xi)) {
        return NaN;
    }
    else {
        return xi > 0 ? 1 : stepAttrs.alpha;
    }
});
const stepConfig = {
    kernelName: Step,
    backendName: 'cpu',
    kernelFunc: step,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function stridedSlice(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask } = attrs;
    assertNotComplex(x, 'stridedSlice');
    const { nonStrided, $begin, $strides, size, newShape, outShape } = sliceInfo(x.shape, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask);
    const $x = reshape({ inputs: { x }, backend, attrs: { shape: newShape } });
    let result;
    if (nonStrided) {
        const sliced = slice({ inputs: { x: $x }, backend, attrs: { begin: $begin, size } });
        result = reshape({ inputs: { x: sliced }, backend, attrs: { shape: outShape } });
        backend.disposeIntermediateTensorInfo(sliced);
    }
    else if (outShape.some(axis => axis === 0)) {
        result = backend.makeTensorInfo(outShape, x.dtype, []);
    }
    else {
        const xBuf = backend.bufferSync($x);
        const outBuf = stridedSliceImpl(outShape, xBuf, $strides, $begin);
        result = backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
    }
    const resultReshaped = reshape({ inputs: { x: result }, backend, attrs: { shape: outShape } });
    backend.disposeIntermediateTensorInfo($x);
    backend.disposeIntermediateTensorInfo(result);
    return resultReshaped;
}
const stridedSliceConfig = {
    kernelName: StridedSlice,
    backendName: 'cpu',
    kernelFunc: stridedSlice
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const tan = unaryKernelFunc(Tan, (xi) => Math.tan(xi));
const tanConfig = {
    kernelName: Tan,
    backendName: 'cpu',
    kernelFunc: tan,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const tanh = unaryKernelFunc(Tanh, (xi) => Math.tanh(xi));
const tanhConfig = {
    kernelName: Tanh,
    backendName: 'cpu',
    kernelFunc: tanh,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function tile(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { reps } = attrs;
    assertNotComplex(x, 'tile');
    const outBuf = tileImpl(backend.bufferSync(x), reps);
    return backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
}
const tileConfig = {
    kernelName: Tile,
    backendName: 'cpu',
    kernelFunc: tile
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function topK(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { k, sorted } = attrs;
    assertNotComplex(x, 'topk');
    const xVals = backend.data.get(x.dataId).values;
    const [allTopKVals, allTopKIndices] = topKImpl(xVals, x.shape, x.dtype, k);
    return [
        backend.makeTensorInfo(allTopKVals.shape, allTopKVals.dtype, allTopKVals.values),
        backend.makeTensorInfo(allTopKIndices.shape, allTopKIndices.dtype, allTopKIndices.values)
    ];
}
const topKConfig = {
    kernelName: TopK,
    backendName: 'cpu',
    kernelFunc: topK
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function unique(args) {
    const { inputs, attrs, backend } = args;
    const { axis } = attrs;
    const { x } = inputs;
    assertNotComplex(x, 'unique');
    const values = backend.data.get(x.dataId).values;
    const { outputValues, outputShape, indices } = uniqueImpl(values, axis, x.shape, x.dtype);
    return [
        backend.makeTensorInfo(outputShape, x.dtype, outputValues),
        backend.makeTensorInfo([indices.length], 'int32', indices),
    ];
}
const uniqueConfig = {
    kernelName: Unique,
    backendName: 'cpu',
    kernelFunc: unique,
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function unpack(args) {
    const { inputs, backend, attrs } = args;
    const { value } = inputs;
    let { axis } = attrs;
    if (axis < 0) {
        axis += value.shape.length;
    }
    const valueRank = value.shape.length;
    const num = value.shape[axis];
    const outShape = new Array(valueRank - 1);
    let outIndex = 0;
    for (let i = 0; i < valueRank; i++) {
        if (i !== axis) {
            outShape[outIndex++] = value.shape[i];
        }
    }
    const begin = new Array(valueRank).fill(0);
    const size = value.shape.slice();
    size[axis] = 1;
    const res = new Array(num);
    for (let i = 0; i < res.length; i++) {
        begin[axis] = i;
        const tempRes = slice({ inputs: { x: value }, backend, attrs: { begin, size } });
        res[i] = reshape({ inputs: { x: tempRes }, backend, attrs: { shape: outShape } });
        backend.disposeIntermediateTensorInfo(tempRes);
    }
    return res;
}
const unpackConfig = {
    kernelName: Unpack,
    backendName: 'cpu',
    kernelFunc: unpack
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
function unsortedSegmentSum(args) {
    const { inputs, backend, attrs } = args;
    const { x, segmentIds } = inputs;
    const { numSegments } = attrs;
    assertNotComplex(x, 'unsortedSegmentSum');
    const xRank = x.shape.length;
    const segmentIdsRank = segmentIds.shape.length;
    const res = [];
    const intermediates = [];
    // Reshape the segment id's so that they can be broadcast with
    // x. The new shape should be [segmentIds.shape, 1, ..., 1]
    const numIters = xRank - segmentIdsRank;
    let $segmentIds = segmentIds;
    for (let i = 0; i < numIters; ++i) {
        const expanded = expandDims({ inputs: { input: $segmentIds }, backend, attrs: { dim: i + 1 } });
        $segmentIds = expanded;
        intermediates.push(expanded);
    }
    for (let i = 0; i < numSegments; ++i) {
        const scalarValue = createScalarValue(i, 'int32');
        const segmentId = backend.makeTensorInfo([], 'int32', scalarValue);
        const mask = equal({ inputs: { a: segmentId, b: $segmentIds }, backend });
        const maskCasted = cast({ inputs: { x: mask }, backend, attrs: { dtype: 'float32' } });
        const mul = multiply({ inputs: { a: maskCasted, b: x }, backend });
        const sumTensorInfo = sum({ inputs: { x: mul }, backend, attrs: { axis: 0, keepDims: false } });
        res.push(sumTensorInfo);
        intermediates.push(segmentId);
        intermediates.push(mask);
        intermediates.push(maskCasted);
        intermediates.push(mul);
        intermediates.push(sumTensorInfo);
    }
    const result = pack({ inputs: res, backend, attrs: { axis: 0 } });
    intermediates.forEach(t => backend.disposeIntermediateTensorInfo(t));
    return result;
}
const unsortedSegmentSumConfig = {
    kernelName: UnsortedSegmentSum,
    backendName: 'cpu',
    kernelFunc: unsortedSegmentSum
};

/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// List all kernel configs here
const kernelConfigs = [
    _fusedMatMulConfig,
    absConfig,
    acosConfig,
    acoshConfig,
    addConfig,
    addNConfig,
    allConfig,
    anyConfig,
    argMaxConfig,
    argMinConfig,
    asinConfig,
    asinhConfig,
    atanConfig,
    atan2Config,
    atanhConfig,
    avgPoolConfig,
    avgPool3DConfig,
    avgPool3DGradConfig,
    avgPoolGradConfig,
    batchMatMulConfig,
    batchNormConfig,
    batchToSpaceNDConfig,
    bincountConfig,
    castConfig,
    ceilConfig,
    clipConfig,
    complexConfig,
    complexAbsConfig,
    concatConfig,
    conv2DBackpropFilterConfig,
    conv2DBackpropInputConfig,
    conv2DConfig,
    conv3DBackpropFilterV2Config,
    conv3DBackpropInputV2Config,
    conv3DConfig,
    cosConfig,
    coshConfig,
    cropAndResizeConfig,
    cumsumConfig,
    denseBincountConfig,
    depthToSpaceConfig,
    depthwiseConv2dNativeConfig,
    depthwiseConv2dNativeBackpropFilterConfig,
    depthwiseConv2dNativeBackpropInputConfig,
    diagConfig,
    dilation2dConfig,
    dilation2dBackpropInputConfig,
    dilation2dBackpropFilterConfig,
    realDivConfig,
    eluConfig,
    eluGradConfig,
    equalConfig,
    erfConfig,
    expConfig,
    expandDimsConfig,
    expm1Config,
    fftConfig,
    fillConfig,
    flipLeftRightConfig,
    floorConfig,
    floorDivConfig,
    fusedConv2DConfig,
    fusedDepthwiseConv2DConfig,
    gatherNdConfig,
    gatherV2Config,
    greaterConfig,
    greaterEqualConfig,
    identityConfig,
    ifftConfig,
    imagConfig,
    isFiniteConfig,
    isInfConfig,
    isNaNConfig,
    leakyReluConfig,
    lessConfig,
    lessEqualConfig,
    linSpaceConfig,
    logConfig,
    log1pConfig,
    logicalAndConfig,
    logicalNotConfig,
    logicalOrConfig,
    lRNConfig,
    lRNGradConfig,
    maximumConfig,
    maxPoolConfig,
    maxPool3DConfig,
    maxPool3DGradConfig,
    maxPoolGradConfig,
    maxPoolWithArgmaxConfig,
    maxConfig,
    meanConfig,
    minConfig,
    minimumConfig,
    mirrorPadConfig,
    modConfig,
    multinomialConfig,
    multiplyConfig,
    negConfig,
    nonMaxSuppressionV3Config,
    nonMaxSuppressionV4Config,
    nonMaxSuppressionV5Config,
    notEqualConfig,
    oneHotConfig,
    onesLikeConfig,
    packConfig,
    padV2Config,
    powConfig,
    preluConfig,
    prodConfig,
    rangeConfig,
    realConfig,
    reciprocalConfig,
    reluConfig,
    relu6Config,
    reshapeConfig,
    resizeBilinearConfig,
    resizeBilinearGradConfig,
    resizeNearestNeighborConfig,
    resizeNearestNeighborGradConfig,
    reverseConfig,
    rotateWithOffsetConfig,
    roundConfig,
    rsqrtConfig,
    scatterNdConfig,
    selectConfig,
    seluConfig,
    sigmoidConfig,
    signConfig,
    sinConfig,
    sinhConfig,
    sliceConfig,
    softmaxConfig,
    softplusConfig,
    spaceToBatchNDConfig,
    sparseToDenseConfig,
    splitVConfig,
    sqrtConfig,
    squareConfig,
    squaredDifferenceConfig,
    stepConfig,
    stridedSliceConfig,
    subConfig,
    sumConfig,
    tanConfig,
    tanhConfig,
    tileConfig,
    topKConfig,
    transposeConfig,
    uniqueConfig,
    unpackConfig,
    unsortedSegmentSumConfig,
    zerosLikeConfig
];
for (const kernelConfig of kernelConfigs) {
    registerKernel(kernelConfig);
}

export { MathBackendCPU, version as version_cpu };
