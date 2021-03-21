import { f as assert, M as Abs, y as sizeFromShape, ap as assertAndGetBroadcastShape, cl as computeStrides, cf as getTypedArrayFromDType, c_ as getBroadcastDims, cS as indexToLoc, cD as locToIndex, c$ as Complex, c3 as makeZerosTypedArray, d0 as Identity, bc as Real, d1 as Cast, d2 as hasEncodingLoss, cE as toTypedArray, J as Add, ay as buffer, cC as getArrayFromDType, ad as Ceil, av as Exp, ax as Expm1, aA as Floor, aC as Greater, aF as Less, aJ as Log, aZ as Maximum, b0 as Minimum, d3 as Multiply, cJ as createScalarValue, aN as Neg, b4 as NotEqual, H as Transpose, cp as computeOutAndReduceShapes, cy as upcastType, ba as Prod, aR as parseAxisParam, cm as getAxesPermutation, cn as getInnerMostAxes, aS as expandShapeToKeepDim, bg as Rsqrt, a7 as Slice, br as SquaredDifference, aQ as Sub, bi as TensorBuffer } from './non_max_suppression_impl-886052e2.js';
import { m as mergeRealAndImagArrays, B as fromUint8ToStringArray, C as isSliceContinous, D as computeFlatOffset, F as fromStringArrayToUint8, G as parseSliceParams, H as assertParamsValid } from './backend_util-5208330b.js';

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
function assertNotComplex(tensor, opName) {
    if (!Array.isArray(tensor)) {
        tensor = [tensor];
    }
    tensor.forEach(t => {
        if (t != null) {
            assert(t.dtype !== 'complex64', () => `${opName} does not support complex64 tensors in the CPU backend.`);
        }
    });
}

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
function simpleAbsImpl(vals) {
    const resultValues = new Float32Array(vals.length);
    for (let i = 0; i < vals.length; ++i) {
        resultValues[i] = Math.abs(vals[i]);
    }
    return resultValues;
}
const abs = (args) => {
    const { x } = args.inputs;
    const cpuBackend = args.backend;
    assertNotComplex(x, 'abs');
    let resultValues = new Float32Array(sizeFromShape(x.shape));
    const values = cpuBackend.data.get(x.dataId).values;
    resultValues = simpleAbsImpl(values);
    return cpuBackend.makeOutput(resultValues, x.shape, 'float32');
};
const absConfig = {
    kernelName: Abs,
    backendName: 'cpu',
    kernelFunc: abs,
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
 * Template that creates implementation for binary ops. Supports broadcast.
 */
function createSimpleBinaryKernelImpl(op) {
    return (aShape, bShape, aVals, bVals, dtype) => {
        const newShape = assertAndGetBroadcastShape(aShape, bShape);
        const resultRank = newShape.length;
        const resultStrides = computeStrides(newShape);
        const resultSize = sizeFromShape(newShape);
        const result = getTypedArrayFromDType(dtype, resultSize);
        const aRank = aShape.length;
        const bRank = bShape.length;
        const aStrides = computeStrides(aShape);
        const bStrides = computeStrides(bShape);
        const aBroadcastDims = getBroadcastDims(aShape, newShape);
        const bBroadcastDims = getBroadcastDims(bShape, newShape);
        if (aBroadcastDims.length + bBroadcastDims.length === 0) {
            for (let i = 0; i < result.length; ++i) {
                result[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
            }
        }
        else {
            for (let i = 0; i < result.length; ++i) {
                const loc = indexToLoc(i, resultRank, resultStrides);
                const aLoc = loc.slice(-aRank);
                aBroadcastDims.forEach(d => aLoc[d] = 0);
                const aIndex = locToIndex(aLoc, aRank, aStrides);
                const bLoc = loc.slice(-bRank);
                bBroadcastDims.forEach(d => bLoc[d] = 0);
                const bIndex = locToIndex(bLoc, bRank, bStrides);
                result[i] = op(aVals[aIndex], bVals[bIndex]);
            }
        }
        return [result, newShape];
    };
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
function complex(args) {
    const { inputs, backend } = args;
    const { real, imag } = inputs;
    const realVals = backend.data.get(real.dataId).values;
    const imagVals = backend.data.get(imag.dataId).values;
    const complexInfo = backend.makeTensorInfo(real.shape, 'complex64');
    const complex = backend.data.get(complexInfo.dataId);
    // The complex tensor owns the underlying real and imag tensorInfos, only the
    // complex tensor tracks refCount, when complexData is disposed the
    // underlying tensorData will be disposed.
    complex.complexTensorInfos = {
        real: backend.makeTensorInfo(real.shape, 'float32', realVals),
        imag: backend.makeTensorInfo(imag.shape, 'float32', imagVals)
    };
    return complexInfo;
}
const complexConfig = {
    kernelName: Complex,
    backendName: 'cpu',
    kernelFunc: complex
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
 * Generates a tensorInfo with all zeros value.
 * @param backend cpu backend.
 * @param shape Shape for the zeros tensor.
 * @param dtype Optional. If set, the result has this dtype.
 */
function zeros(backend, shape, dtype = 'float32') {
    if (dtype === 'complex64') {
        const real = zeros(backend, shape, 'float32');
        const imag = zeros(backend, shape, 'float32');
        return complex({ inputs: { real, imag }, backend });
    }
    const values = makeZerosTypedArray(sizeFromShape(shape), dtype);
    return backend.makeTensorInfo(shape, dtype, values);
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
function identity(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    backend.incRef(x.dataId);
    return { dataId: x.dataId, shape: x.shape, dtype: x.dtype };
}
const identityConfig = {
    kernelName: Identity,
    backendName: 'cpu',
    kernelFunc: identity
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
function real(args) {
    const { inputs, backend } = args;
    const { input } = inputs;
    const real = backend.data.get(input.dataId).complexTensorInfos.real;
    const realVal = backend.data.get(real.dataId).values;
    // When complex tensor is disposed, its underlying parts will be disposed too.
    // Make new tensor out of the real value of the complex. This makes sure the
    // value is still accessible even if complex tensor is disposed.
    return backend.makeTensorInfo(real.shape, real.dtype, realVal);
}
const realConfig = {
    kernelName: Real,
    backendName: 'cpu',
    kernelFunc: real
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
function cast(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { dtype } = attrs;
    // Casting to complex64.
    if (dtype === 'complex64') {
        if (x.dtype === 'complex64') {
            return identity({ inputs: { x }, backend });
        }
        const zerosTensorInfo = zeros(backend, x.shape, x.dtype);
        const floatX = cast({ inputs: { x }, backend, attrs: { dtype: 'float32' } });
        const result = complex({ inputs: { real: floatX, imag: zerosTensorInfo }, backend });
        backend.disposeIntermediateTensorInfo(zerosTensorInfo);
        backend.disposeIntermediateTensorInfo(floatX);
        return result;
    }
    // Casting from complex64
    if (x.dtype === 'complex64') {
        const realPart = real({ inputs: { input: x }, backend });
        const result = cast({ inputs: { x: realPart }, backend, attrs: { dtype } });
        backend.disposeIntermediateTensorInfo(realPart);
        return result;
    }
    if (!hasEncodingLoss(x.dtype, dtype)) {
        // We don't change the underlying data, since we cast to higher
        // precision.
        const result = identity({ inputs: { x }, backend });
        return { dataId: result.dataId, shape: result.shape, dtype };
    }
    if (dtype === 'int32') {
        const values = backend.data.get(x.dataId).values;
        const resultValues = Int32Array.from(values);
        return backend.makeTensorInfo(x.shape, 'int32', resultValues);
    }
    if (dtype === 'bool') {
        // This is essentially the result of notEqual(x, 0). We avoid using
        // kernel notEqual to avoid circular dependency, i.e. binary_utils ->
        // cast -> notEqual -> binary_utils.
        const xVals = backend.data.get(x.dataId).values;
        const zero = toTypedArray([0], x.dtype);
        const [resultData, resultShape] = createSimpleBinaryKernelImpl((a, b) => (a !== b) ? 1 : 0)(x.shape, [], xVals, zero, 'bool');
        return backend.makeTensorInfo(resultShape, 'bool', resultData);
    }
    throw new Error(`Error in Cast: failed to cast ${x.dtype} to ${dtype}`);
}
const castConfig = {
    kernelName: Cast,
    backendName: 'cpu',
    kernelFunc: cast
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
 * Template that creates a `KernelFunc` for binary ops.
 * @param name Kernel name.
 * @param binaryKernelImpl A `SimpleBinaryKernelImpl` for the kernel.
 * @param binaryKernelComplexImpl Optional. If exists, represents a
 *     `ComplexBinaryKernelImpl` for the kernel, will be used when input dtype
 *     is `complex64`.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the first input. This is mainly used in
 *     comparison kernels, such as Equal, Less, Greater, etc.
 */
function binaryKernelFunc(name, simpleImpl, complexImpl, dtype) {
    if (complexImpl == null) {
        return ({ inputs, backend }) => {
            const { a, b } = inputs;
            const cpuBackend = backend;
            assertNotComplex([a, b], name);
            const aVals = cpuBackend.data.get(a.dataId).values;
            const bVals = cpuBackend.data.get(b.dataId).values;
            const $dtype = dtype || a.dtype;
            const [resultData, resultShape] = simpleImpl(a.shape, b.shape, aVals, bVals, $dtype);
            return cpuBackend.makeTensorInfo(resultShape, $dtype, resultData);
        };
    }
    return ({ inputs, backend }) => {
        const { a, b } = inputs;
        const cpuBackend = backend;
        if (a.dtype === 'complex64' || b.dtype === 'complex64') {
            const $aComplex = cast({ inputs: { x: a }, backend: cpuBackend, attrs: { dtype: 'complex64' } });
            const $aComplexVals = cpuBackend.data.get($aComplex.dataId);
            const aReal = $aComplexVals.complexTensorInfos.real;
            const aImag = $aComplexVals.complexTensorInfos.imag;
            const aRealVals = cpuBackend.data.get(aReal.dataId).values;
            const aImagVals = cpuBackend.data.get(aImag.dataId).values;
            const $bComplex = cast({ inputs: { x: b }, backend: cpuBackend, attrs: { dtype: 'complex64' } });
            const $bComplexVals = cpuBackend.data.get($bComplex.dataId);
            const bReal = $bComplexVals.complexTensorInfos.real;
            const bImag = $bComplexVals.complexTensorInfos.imag;
            const bRealVals = cpuBackend.data.get(bReal.dataId).values;
            const bImagVals = cpuBackend.data.get(bImag.dataId).values;
            const [resultRealData, resultImagData, resultShape] = complexImpl(a.shape, b.shape, aRealVals, aImagVals, bRealVals, bImagVals);
            const resultReal = cpuBackend.makeTensorInfo(resultShape, 'float32', resultRealData);
            const resultImag = cpuBackend.makeTensorInfo(resultShape, 'float32', resultImagData);
            const result = complex({ inputs: { real: resultReal, imag: resultImag }, backend: cpuBackend });
            cpuBackend.disposeIntermediateTensorInfo($aComplex);
            cpuBackend.disposeIntermediateTensorInfo($bComplex);
            cpuBackend.disposeIntermediateTensorInfo(resultReal);
            cpuBackend.disposeIntermediateTensorInfo(resultImag);
            return result;
        }
        else {
            const aVals = cpuBackend.data.get(a.dataId).values;
            const bVals = cpuBackend.data.get(b.dataId).values;
            const $dtype = dtype || a.dtype;
            const [resultData, resultShape] = simpleImpl(a.shape, b.shape, aVals, bVals, $dtype);
            return cpuBackend.makeTensorInfo(resultShape, $dtype, resultData);
        }
    };
}
/**
 * Template that creates the complex type implementation for binary ops.
 * Supports broadcast.
 */
function createComplexBinaryKernelImpl(op) {
    return (aShape, bShape, aRealVals, aImagVals, bRealVals, bImagVals) => {
        const resultShape = assertAndGetBroadcastShape(aShape, bShape);
        const resultSize = sizeFromShape(resultShape);
        const resultRank = resultShape.length;
        const resultStrides = computeStrides(resultShape);
        const resultRealVals = getTypedArrayFromDType('float32', resultSize);
        const resultImagVals = getTypedArrayFromDType('float32', resultSize);
        const aBroadcastDims = getBroadcastDims(aShape, resultShape);
        const bBroadcastDims = getBroadcastDims(bShape, resultShape);
        const aVals = mergeRealAndImagArrays(aRealVals, aImagVals);
        const bVals = mergeRealAndImagArrays(bRealVals, bImagVals);
        const aRank = aShape.length;
        const aStrides = computeStrides(aShape);
        const bRank = bShape.length;
        const bStrides = computeStrides(bShape);
        if (aBroadcastDims.length + bBroadcastDims.length === 0) {
            for (let i = 0; i < resultRealVals.length; i++) {
                const aIdx = i % aVals.length;
                const bIdx = i % bVals.length;
                const result = op(aVals[aIdx * 2], aVals[aIdx * 2 + 1], bVals[bIdx * 2], bVals[bIdx * 2 + 1]);
                resultRealVals[i] = result.real;
                resultImagVals[i] = result.imag;
            }
        }
        else {
            for (let i = 0; i < resultRealVals.length; i++) {
                const loc = indexToLoc(i, resultRank, resultStrides);
                const aLoc = loc.slice(-aRank);
                aBroadcastDims.forEach(d => aLoc[d] = 0);
                const aIndex = locToIndex(aLoc, aRank, aStrides);
                const bLoc = loc.slice(-bRank);
                bBroadcastDims.forEach(d => bLoc[d] = 0);
                const bIndex = locToIndex(bLoc, bRank, bStrides);
                const opResult = op(aVals[aIndex * 2], aVals[aIndex * 2 + 1], bVals[bIndex * 2], bVals[bIndex * 2 + 1]);
                resultRealVals[i] = opResult.real;
                resultImagVals[i] = opResult.imag;
            }
        }
        return [resultRealVals, resultImagVals, resultShape];
    };
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
const addImpl = createSimpleBinaryKernelImpl(((a, b) => a + b));
const addComplexImpl = createComplexBinaryKernelImpl(((aReal, aImag, bReal, bImag) => {
    return { real: aReal + bReal, imag: aImag + bImag };
}));
const add = binaryKernelFunc(Add, addImpl, addComplexImpl);
const addConfig = {
    kernelName: Add,
    backendName: 'cpu',
    kernelFunc: add
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
function bincountImpl(xVals, weightsVals, weightsDtype, weightsShape, size) {
    const weightsSize = sizeFromShape(weightsShape);
    const outVals = makeZerosTypedArray(size, weightsDtype);
    for (let i = 0; i < xVals.length; i++) {
        const value = xVals[i];
        if (value < 0) {
            throw new Error('Input x must be non-negative!');
        }
        if (value >= size) {
            continue;
        }
        if (weightsSize > 0) {
            outVals[value] += weightsVals[i];
        }
        else {
            outVals[value] += 1;
        }
    }
    return outVals;
}
function bincountReduceImpl(xBuf, weightsBuf, size, binaryOutput = false) {
    const numRows = xBuf.shape[0];
    const numCols = xBuf.shape[1];
    const outBuf = buffer([numRows, size], weightsBuf.dtype);
    for (let i = 0; i < numRows; i++) {
        for (let j = 0; j < numCols; j++) {
            const value = xBuf.get(i, j);
            if (value < 0) {
                throw new Error('Input x must be non-negative!');
            }
            if (value >= size) {
                continue;
            }
            if (binaryOutput) {
                outBuf.set(1, i, value);
            }
            else {
                if (weightsBuf.size > 0) {
                    outBuf.set(outBuf.get(i, value) + weightsBuf.get(i, j), i, value);
                }
                else {
                    outBuf.set(outBuf.get(i, value) + 1, i, value);
                }
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
/**
 * Template that creates implementation for unary op.
 */
function createSimpleUnaryImpl(op) {
    return (values, dtype, attrs) => {
        const newValues = getTypedArrayFromDType(dtype, values.length);
        for (let i = 0; i < values.length; ++i) {
            newValues[i] = op(values[i], attrs);
        }
        return newValues;
    };
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
/**
 * Template that creates a `KernelFunc` for unary ops.
 * @param name Kernel name.
 * @param op A `SimpleUnaryOperation` for the kernel.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the input. This is mainly used in certain
 *     kernels that return bool type, such as isFinite, isInf, etc.
 */
function unaryKernelFunc(name, op, dtype) {
    return ({ inputs, attrs, backend }) => {
        const { x } = inputs;
        assertNotComplex(x, name);
        if (x.dtype === 'string' || dtype === 'string') {
            throw new Error('unaryKernelFunc does not support string input/output');
        }
        const cpuBackend = backend;
        const values = cpuBackend.data.get(x.dataId).values;
        const xSize = sizeFromShape(x.shape);
        const $dtype = dtype || x.dtype;
        const newValues = getArrayFromDType($dtype, xSize);
        for (let i = 0; i < xSize; ++i) {
            newValues[i] = op(values[i], attrs);
        }
        return cpuBackend.makeTensorInfo(x.shape, $dtype, newValues);
    };
}
/**
 * Template that creates a `KernelFunc` for unary ops from the given
 * `SimpleUnaryImpl`..
 * @param name Kernel name.
 * @param unaryImpl A `SimpleUnaryImpl` that implements the op.
 * @param dtype Optional. If set, the result has this dtype. Otherwise, the
 *     result has the same dtype as the input. This is mainly used in certain
 *     kernels that return bool type, such as isFinite, isInf, etc.
 */
function unaryKernelFuncFromImpl(name, unaryImpl, dtype) {
    return ({ inputs, attrs, backend }) => {
        const { x } = inputs;
        assertNotComplex(x, name);
        if (x.dtype === 'string' || dtype === 'string') {
            throw new Error('unaryKernelFunc does not support string input/output');
        }
        const cpuBackend = backend;
        const values = cpuBackend.data.get(x.dataId).values;
        const $dtype = dtype || x.dtype;
        const newValues = unaryImpl(values, $dtype, attrs);
        return cpuBackend.makeTensorInfo(x.shape, $dtype, newValues);
    };
}

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
const ceilImpl = createSimpleUnaryImpl((xi) => Math.ceil(xi));
const ceil = unaryKernelFuncFromImpl(Ceil, ceilImpl);
const ceilConfig = {
    kernelName: Ceil,
    backendName: 'cpu',
    kernelFunc: ceil,
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
function concatImpl(inputs, outShape, dtype, simplyConcat) {
    const outVals = getArrayFromDType(dtype, sizeFromShape(outShape));
    if (simplyConcat && dtype !== 'string') {
        // Use built-in TypedArray.set() method for speed.
        let offset = 0;
        inputs.forEach(input => {
            const size = sizeFromShape(input.shape);
            outVals.set(input.vals, offset);
            offset += size;
        });
    }
    else {
        let colOffset = 0;
        inputs.forEach(input => {
            const decodedData = dtype === 'string' ?
                fromUint8ToStringArray(input.vals) :
                input.vals;
            let tIdx = 0;
            for (let row = 0; row < input.shape[0]; ++row) {
                const resIdx = row * outShape[1] + colOffset;
                for (let col = 0; col < input.shape[1]; ++col) {
                    outVals[resIdx + col] = decodedData[tIdx++];
                }
            }
            colOffset += input.shape[1];
        });
    }
    return outVals;
}

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
const expImpl = createSimpleUnaryImpl((xi) => Math.exp(xi));
const exp = unaryKernelFuncFromImpl(Exp, expImpl);
const expConfig = {
    kernelName: Exp,
    backendName: 'cpu',
    kernelFunc: exp,
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
const expm1Impl = createSimpleUnaryImpl((xi) => Math.expm1(xi));
const expm1 = unaryKernelFuncFromImpl(Expm1, expm1Impl);
const expm1Config = {
    kernelName: Expm1,
    backendName: 'cpu',
    kernelFunc: expm1,
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
const floorImpl = createSimpleUnaryImpl((xi) => Math.floor(xi));
const floor = unaryKernelFuncFromImpl(Floor, floorImpl);
const floorConfig = {
    kernelName: Floor,
    backendName: 'cpu',
    kernelFunc: floor,
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
function gatherV2Impl(xBuf, indicesBuf, flattenOutputShape) {
    const outBuf = buffer(flattenOutputShape, xBuf.dtype);
    for (let i = 0; i < outBuf.size; ++i) {
        const newLoc = outBuf.indexToLoc(i);
        const originalLoc = newLoc.slice();
        const batchIdx = originalLoc[0];
        const indicesIdx = originalLoc[2];
        const indicesIndex = indicesBuf.locToIndex([batchIdx, indicesIdx]);
        originalLoc[2] = indicesBuf.values[indicesIndex];
        const originalIndex = xBuf.locToIndex(originalLoc);
        outBuf.values[i] = xBuf.values[originalIndex];
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
const greaterImpl = createSimpleBinaryKernelImpl((a, b) => (a > b) ? 1 : 0);
const greater = binaryKernelFunc(Greater, greaterImpl, null /* complexImpl */, 'bool');
const greaterConfig = {
    kernelName: Greater,
    backendName: 'cpu',
    kernelFunc: greater
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
const lessImpl = createSimpleBinaryKernelImpl((a, b) => (a < b) ? 1 : 0);
const less = binaryKernelFunc(Less, lessImpl, null /* complexImpl */, 'bool');
const lessConfig = {
    kernelName: Less,
    backendName: 'cpu',
    kernelFunc: less
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
function linSpaceImpl(start, stop, num) {
    const step = (stop - start) / (num - 1);
    const values = makeZerosTypedArray(num, 'float32');
    values[0] = start;
    for (let i = 1; i < values.length; i++) {
        values[i] = values[i - 1] + step;
    }
    return values;
}

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
const logImpl = createSimpleUnaryImpl((xi) => Math.log(xi));
const log = unaryKernelFuncFromImpl(Log, logImpl);
const logConfig = {
    kernelName: Log,
    backendName: 'cpu',
    kernelFunc: log,
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
function maxImpl(aVals, reduceSize, outShape, dtype) {
    const vals = getTypedArrayFromDType(dtype, sizeFromShape(outShape));
    for (let i = 0; i < vals.length; ++i) {
        const offset = i * reduceSize;
        let max = aVals[offset];
        for (let j = 0; j < reduceSize; ++j) {
            const value = aVals[offset + j];
            if (value > max) {
                max = value;
            }
        }
        vals[i] = max;
    }
    return vals;
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
const maximumImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => Math.max(aValue, bValue)));
const maximum = binaryKernelFunc(Maximum, maximumImpl);
const maximumConfig = {
    kernelName: Maximum,
    backendName: 'cpu',
    kernelFunc: maximum
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
const minimumImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => Math.min(aValue, bValue)));
const minimum = binaryKernelFunc(Minimum, minimumImpl);
const minimumConfig = {
    kernelName: Minimum,
    backendName: 'cpu',
    kernelFunc: minimum
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
const multiplyImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => aValue * bValue));
const multiplyComplexImpl = createComplexBinaryKernelImpl(((aReal, aImag, bReal, bImag) => {
    return {
        real: aReal * bReal - aImag * bImag,
        imag: aReal * bImag + aImag * bReal
    };
}));
const multiply = binaryKernelFunc(Multiply, multiplyImpl, multiplyComplexImpl);
const multiplyConfig = {
    kernelName: Multiply,
    backendName: 'cpu',
    kernelFunc: multiply
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
function negImpl(xVals, xShape, xDtype) {
    const minusOne = createScalarValue(-1, xDtype);
    return multiplyImpl([], xShape, minusOne, xVals, xDtype);
}
function neg(args) {
    const { inputs, backend } = args;
    const { x } = inputs;
    assertNotComplex(x, 'neg');
    const xVals = backend.data.get(x.dataId).values;
    const [res, newShape] = negImpl(xVals, x.shape, x.dtype);
    return backend.makeTensorInfo(newShape, x.dtype, res);
}
const negConfig = {
    kernelName: Neg,
    backendName: 'cpu',
    kernelFunc: neg
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
const notEqualImpl = createSimpleBinaryKernelImpl(((a, b) => (a !== b) ? 1 : 0));
const notEqual = binaryKernelFunc(NotEqual, notEqualImpl, null /* complexOp */, 'bool');
const notEqualConfig = {
    kernelName: NotEqual,
    backendName: 'cpu',
    kernelFunc: notEqual
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
function transposeImpl(xVals, xShape, dtype, perm, newShape) {
    const xRank = xShape.length;
    const xSize = sizeFromShape(xShape);
    const xStrides = computeStrides(xShape);
    const newStrides = computeStrides(newShape);
    const result = getTypedArrayFromDType(dtype, sizeFromShape(newShape));
    for (let i = 0; i < xSize; ++i) {
        const loc = indexToLoc(i, xRank, xStrides);
        // Permute location.
        const newLoc = new Array(loc.length);
        for (let i = 0; i < newLoc.length; i++) {
            newLoc[i] = loc[perm[i]];
        }
        const newIndex = locToIndex(newLoc, xRank, newStrides);
        result[newIndex] = xVals[i];
    }
    return result;
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
function transpose(args) {
    const { inputs, attrs, backend } = args;
    const { x } = inputs;
    const { perm } = attrs;
    assertNotComplex(x, 'transpose');
    const xRank = x.shape.length;
    const newShape = new Array(xRank);
    for (let i = 0; i < newShape.length; i++) {
        newShape[i] = x.shape[perm[i]];
    }
    const values = backend.data.get(x.dataId).values;
    const result = transposeImpl(values, x.shape, x.dtype, perm, newShape);
    const dataId = backend.write(result, newShape, x.dtype);
    return { dataId, shape: newShape, dtype: x.dtype };
}
const transposeConfig = {
    kernelName: Transpose,
    backendName: 'cpu',
    kernelFunc: transpose
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
function prodImpl(xShape, xDtype, xVals, reductionAxes) {
    const [outShape, reduceShape] = computeOutAndReduceShapes(xShape, reductionAxes);
    const outDtype = upcastType(xDtype, 'int32');
    const outVals = makeZerosTypedArray(sizeFromShape(outShape), outDtype);
    const reduceSize = sizeFromShape(reduceShape);
    for (let i = 0; i < outVals.length; ++i) {
        const offset = i * reduceSize;
        let prod = 1;
        for (let j = 0; j < reduceSize; ++j) {
            prod *= xVals[offset + j];
        }
        outVals[i] = prod;
    }
    return { outVals, outShape, outDtype };
}
function prod(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { axis, keepDims } = attrs;
    assertNotComplex(x, 'prod');
    const xRank = x.shape.length;
    const axes = parseAxisParam(axis, x.shape);
    const permutation = getAxesPermutation(axes, xRank);
    let reductionAxes = axes;
    let permutedX = x;
    const intermediateTensorInfos = [];
    if (permutation != null) {
        permutedX = transpose({ inputs: { x }, backend, attrs: { perm: permutation } });
        intermediateTensorInfos.push(permutedX);
        reductionAxes = getInnerMostAxes(reductionAxes.length, xRank);
    }
    const xVals = backend.data.get(permutedX.dataId).values;
    const { outVals, outShape, outDtype } = prodImpl(permutedX.shape, permutedX.dtype, xVals, reductionAxes);
    let resultShape = outShape;
    if (keepDims) {
        resultShape = expandShapeToKeepDim(outShape, axes);
    }
    intermediateTensorInfos.forEach(t => backend.disposeIntermediateTensorInfo(t));
    return backend.makeTensorInfo(resultShape, outDtype, outVals);
}
const prodConfig = {
    kernelName: Prod,
    backendName: 'cpu',
    kernelFunc: prod
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
function rangeImpl(start, stop, step, dtype) {
    const sameStartStop = start === stop;
    const increasingRangeNegativeStep = start < stop && step < 0;
    const decreasingRangePositiveStep = stop < start && step > 1;
    if (sameStartStop || increasingRangeNegativeStep ||
        decreasingRangePositiveStep) {
        return makeZerosTypedArray(0, dtype);
    }
    const numElements = Math.abs(Math.ceil((stop - start) / step));
    const values = makeZerosTypedArray(numElements, dtype);
    if (stop < start && step === 1) {
        // Auto adjust the step's sign if it hasn't been set
        // (or was set to 1)
        step = -1;
    }
    values[0] = start;
    for (let i = 1; i < values.length; i++) {
        values[i] = values[i - 1] + step;
    }
    return values;
}

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
const rsqrtImpl = createSimpleUnaryImpl((xi) => 1 / Math.sqrt(xi));
const rsqrt = unaryKernelFuncFromImpl(Rsqrt, rsqrtImpl);
const rsqrtConfig = {
    kernelName: Rsqrt,
    backendName: 'cpu',
    kernelFunc: rsqrt,
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
function sliceImpl(vals, begin, size, shape, dtype) {
    const isContinous = isSliceContinous(shape, begin, size);
    const length = sizeFromShape(size);
    const xStrides = computeStrides(shape);
    if (isContinous) {
        const flatOffset = computeFlatOffset(begin, xStrides);
        if (dtype === 'string') {
            return vals.slice(flatOffset, flatOffset + length);
        }
        return vals.subarray(flatOffset, flatOffset + length);
    }
    const decodedData = dtype === 'string' ?
        fromUint8ToStringArray(vals) :
        vals;
    const inBuf = buffer(shape, dtype, decodedData);
    const outBuf = buffer(size, dtype);
    for (let i = 0; i < outBuf.size; ++i) {
        const outLoc = outBuf.indexToLoc(i);
        const inLoc = outLoc.map((idx, j) => idx + begin[j]);
        outBuf.set(inBuf.get(...inLoc), ...outLoc);
    }
    if (dtype === 'string') {
        return fromStringArrayToUint8(outBuf.values);
    }
    return outBuf.values;
}
function slice(args) {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { begin, size } = attrs;
    assertNotComplex(x, 'slice');
    const [$begin, $size] = parseSliceParams(x, begin, size);
    assertParamsValid(x, $begin, $size);
    const vals = backend.data.get(x.dataId).values;
    const outVals = sliceImpl(vals, $begin, $size, x.shape, x.dtype);
    return backend.makeTensorInfo($size, x.dtype, outVals);
}
const sliceConfig = {
    kernelName: Slice,
    backendName: 'cpu',
    kernelFunc: slice
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
const squaredDifferenceImpl = createSimpleBinaryKernelImpl(((a, b) => {
    const diff = a - b;
    return diff * diff;
}));
const squaredDifference = binaryKernelFunc(SquaredDifference, squaredDifferenceImpl);
const squaredDifferenceConfig = {
    kernelName: SquaredDifference,
    backendName: 'cpu',
    kernelFunc: squaredDifference
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
function stridedSliceImpl(outShape, xBuf, strides, begin) {
    const outBuf = buffer(outShape, xBuf.dtype);
    for (let i = 0; i < outBuf.size; i++) {
        const loc = outBuf.indexToLoc(i);
        const newLoc = new Array(loc.length);
        for (let j = 0; j < newLoc.length; j++) {
            newLoc[j] = loc[j] * strides[j] + begin[j];
        }
        outBuf.set(xBuf.get(...newLoc), ...loc);
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
const subImpl = createSimpleBinaryKernelImpl(((aValue, bValue) => aValue - bValue));
const subComplexImpl = createComplexBinaryKernelImpl(((aReal, aImag, bReal, bImag) => {
    return { real: aReal - bReal, imag: aImag - bImag };
}));
const sub = binaryKernelFunc(Sub, subImpl, subComplexImpl);
const subConfig = {
    kernelName: Sub,
    backendName: 'cpu',
    kernelFunc: sub
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
/**
 * An implementation of the tile kernel shared between webgl and cpu for string
 * tensors only.
 */
function tileImpl(xBuf, reps) {
    const newShape = new Array(xBuf.rank);
    for (let i = 0; i < newShape.length; i++) {
        newShape[i] = xBuf.shape[i] * reps[i];
    }
    const result = buffer(newShape, xBuf.dtype);
    for (let i = 0; i < result.values.length; ++i) {
        const newLoc = result.indexToLoc(i);
        const originalLoc = new Array(xBuf.rank);
        for (let j = 0; j < originalLoc.length; j++) {
            originalLoc[j] = newLoc[j] % xBuf.shape[j];
        }
        const originalIndex = xBuf.locToIndex(originalLoc);
        result.values[i] = xBuf.values[originalIndex];
    }
    return result;
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
function topKImpl(x, xShape, xDtype, k, sorted) {
    // Reshape into a 2d tensor [batch, lastDim] and compute topk along lastDim.
    const lastDim = xShape[xShape.length - 1];
    const [batch, size] = [x.length / lastDim, lastDim];
    const allTopKVals = getTypedArrayFromDType(xDtype, batch * k);
    const allTopKIndices = getTypedArrayFromDType('int32', batch * k);
    for (let b = 0; b < batch; b++) {
        const offset = b * size;
        const vals = x.subarray(offset, offset + size);
        const valAndInd = [];
        for (let i = 0; i < vals.length; i++) {
            valAndInd.push({ value: vals[i], index: i });
        }
        valAndInd.sort((a, b) => b.value - a.value);
        const outOffset = b * k;
        const topKVals = allTopKVals.subarray(outOffset, outOffset + k);
        const topKIndices = allTopKIndices.subarray(outOffset, outOffset + k);
        for (let i = 0; i < k; i++) {
            topKVals[i] = valAndInd[i].value;
            topKIndices[i] = valAndInd[i].index;
        }
    }
    // Reshape back to the original input shape, except that the last
    // dimension is k.
    const outputShape = xShape.slice();
    outputShape[outputShape.length - 1] = k;
    return [
        buffer(outputShape, xDtype, allTopKVals),
        buffer(outputShape, 'int32', allTopKIndices)
    ];
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
function uniqueImpl(values, axis, shape, dtype) {
    // Normalize and validate axis.
    const $axis = parseAxisParam(axis, shape)[0];
    // Calculate the new shape that is suitable for extracting data along the
    // given axis.
    //
    // The rank is 3.
    // The size of the 1st dimension is the size of all the axes < the given axis.
    // The size of the 2nd dimension is the same as the size of the given axis.
    // The size of the 3rd dimension is the size of all the axes > the given axis.
    //
    // For example, for a 4D tensor with shape=[2, 3, 5, 4] and axis=2, the
    // newShape would be: [2*3, 5, 4].
    //
    // Note that this is not the final output shape. This will be the shape for an
    // intermediate TensorBuffer (see inputBuffer below) to allow us to extract
    // values along the given axis. To demonstrate how it works, consider the
    // following example:
    //
    // Input: a 3D tensor, with shape [1, 2, 3]
    // [
    //   [
    //      [1,2,3],
    //      [4,5,6]
    //   ]
    // ]
    // Axis: 2 (the last axis).
    // Along axis 2, we expect to extract 3 tensors: [1,4], [2,5], [3,6].
    //
    // For this example, newShape would be: [2, 3, 1], where 2 is calculated from
    // 1*2. The re-shaped data would look like:
    //
    // [
    //   [
    //     [1], [2], [3]
    //   ],
    //   [
    //     [4], [5], [6]
    //   ]
    // ]
    //
    // Then, we can construct a 3-level nested loop by the following dimension
    // order to extract the values along the axis (dimension1):
    // i: dimension1       // 0,1,2 (newShape[1])
    //   m: dimension0     // 0,1   (newShape[0])
    //     n: dimension2   // 0     (newShape[2])
    //
    //                       m, i, n
    //                      ---------
    // Iteration 0: data at [0, 0, 0] => "1"
    // Iteration 1: data at [1, 0, 0] => "4"
    // We got [1,4].
    // Iteration 2: data at [0, 1, 0] => "2"
    // Iteration 3: data at [1, 1, 0] => "5"
    // We got [2,5].
    // Iteration 4: data at [0, 2, 0] => "3"
    // Iteration 5: data at [1, 2, 0] => "6"
    // We got [3,6].
    const newShape = [1, shape[0], 1];
    for (let i = 0; i < $axis; i++) {
        newShape[0] *= shape[i];
    }
    newShape[1] = shape[$axis];
    for (let i = $axis + 1; i < shape.length; i++) {
        newShape[2] *= shape[i];
    }
    // A map from unique elements (their string representations) to their values
    // in "indices" (below).
    const uniqueElements = {};
    // The indices of each unique element in the original tensor along the given
    // axis. It is 1D and has the same size as the given axis.
    const indices = new Int32Array(shape[$axis]);
    // Create a buffer so we can easily extract value at a given location.
    const inputBuffer = new TensorBuffer(newShape, dtype, values);
    // The indices along the given axis that have unique elements. This is a
    // de-duped version of "indices" above.
    const uniqueIndices = [];
    const is1DTensor = newShape[0] === 1 && newShape[2] === 1;
    for (let i = 0; i < shape[$axis]; i++) {
        // Extract values along the axis.
        let element;
        if (is1DTensor) {
            // Fast path for 1D tensor input.
            element = values[i].toString();
        }
        else {
            const axisValues = [];
            for (let m = 0; m < newShape[0]; m++) {
                for (let n = 0; n < newShape[2]; n++) {
                    axisValues.push(inputBuffer.get(m, i, n));
                }
            }
            element = axisValues.join(',');
        }
        // Dedup and update various indices.
        if (uniqueElements[element] !== undefined) {
            indices[i] = uniqueElements[element];
        }
        else {
            const uniqueIndex = Object.keys(uniqueElements).length;
            uniqueElements[element] = uniqueIndex;
            indices[i] = uniqueIndex;
            uniqueIndices.push(i);
        }
    }
    // Now we know where each of the unique elements are located along the axis
    // (uniqueIndices). Extract them from input buffer and store them in the
    // output buffer.
    const outputTmpShape = newShape.slice();
    outputTmpShape[1] = Object.keys(uniqueElements).length;
    const outputBuffer = new TensorBuffer(outputTmpShape, dtype);
    uniqueIndices.forEach((uniqueElementIndex, i) => {
        for (let m = 0; m < newShape[0]; m++) {
            for (let n = 0; n < newShape[2]; n++) {
                outputBuffer.set(inputBuffer.get(m, uniqueElementIndex, n), m, i, n);
            }
        }
    });
    // The output shape can be calculated from the input shape with the size of
    // the given axis replaced by the number of unique elements along that axis.
    const outputShape = shape.slice();
    outputShape[$axis] = outputTmpShape[1];
    return {
        outputValues: outputBuffer.values,
        outputShape,
        indices,
    };
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

var shared = /*#__PURE__*/Object.freeze({
    __proto__: null,
    simpleAbsImpl: simpleAbsImpl,
    addImpl: addImpl,
    bincountImpl: bincountImpl,
    bincountReduceImpl: bincountReduceImpl,
    ceilImpl: ceilImpl,
    concatImpl: concatImpl,
    expImpl: expImpl,
    expm1Impl: expm1Impl,
    floorImpl: floorImpl,
    gatherV2Impl: gatherV2Impl,
    greaterImpl: greaterImpl,
    lessImpl: lessImpl,
    linSpaceImpl: linSpaceImpl,
    logImpl: logImpl,
    maxImpl: maxImpl,
    maximumImpl: maximumImpl,
    minimumImpl: minimumImpl,
    multiplyImpl: multiplyImpl,
    negImpl: negImpl,
    notEqualImpl: notEqualImpl,
    prodImpl: prodImpl,
    rangeImpl: rangeImpl,
    rsqrtImpl: rsqrtImpl,
    sliceImpl: sliceImpl,
    squaredDifferenceImpl: squaredDifferenceImpl,
    stridedSliceImpl: stridedSliceImpl,
    subImpl: subImpl,
    tileImpl: tileImpl,
    topKImpl: topKImpl,
    transposeImpl: transposeImpl,
    uniqueImpl: uniqueImpl
});

export { uniqueImpl as A, absConfig as B, addConfig as C, castConfig as D, ceilConfig as E, complexConfig as F, expConfig as G, expm1Config as H, floorConfig as I, greaterConfig as J, identityConfig as K, lessConfig as L, logConfig as M, maximumConfig as N, minimumConfig as O, multiplyConfig as P, negConfig as Q, notEqualConfig as R, prodConfig as S, realConfig as T, rsqrtConfig as U, sliceConfig as V, squaredDifferenceConfig as W, subConfig as X, transposeConfig as Y, shared as Z, assertNotComplex as a, add as b, createSimpleBinaryKernelImpl as c, binaryKernelFunc as d, bincountImpl as e, complex as f, concatImpl as g, bincountReduceImpl as h, identity as i, sub as j, gatherV2Impl as k, linSpaceImpl as l, multiply as m, transposeImpl as n, maxImpl as o, cast as p, exp as q, real as r, slice as s, transpose as t, unaryKernelFunc as u, rangeImpl as v, stridedSliceImpl as w, tileImpl as x, topKImpl as y, zeros as z };
