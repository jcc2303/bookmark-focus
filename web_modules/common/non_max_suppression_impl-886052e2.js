import { p as process } from './process-2545f00a.js';

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
const EPSILON_FLOAT32 = 1e-7;
const EPSILON_FLOAT16 = 1e-4;
/** Convenient class for storing tensor-related data. */
class DataStorage {
    constructor(backend, dataMover) {
        this.backend = backend;
        this.dataMover = dataMover;
        this.data = new WeakMap();
        this.dataIdsCount = 0;
    }
    get(dataId) {
        if (!this.data.has(dataId)) {
            this.dataMover.moveData(this.backend, dataId);
        }
        return this.data.get(dataId);
    }
    set(dataId, value) {
        this.dataIdsCount++;
        this.data.set(dataId, value);
    }
    has(dataId) {
        return this.data.has(dataId);
    }
    delete(dataId) {
        this.dataIdsCount--;
        return this.data.delete(dataId);
    }
    numDataIds() {
        return this.dataIdsCount;
    }
}
/**
 * The interface that defines the kernels that should be implemented when
 * adding a new backend. New backends don't need to implement every one of the
 * methods, this can be done gradually (throw an error for unimplemented
 * methods).
 */
class KernelBackend {
    /**
     * Decrease the complex ref count for the dataId, this is useful for WebGL
     * backend to keep the real and imag components of the complex tensor in sync
     * with the engine. WASM and node do not have internal ref count, they will
     * use on the default implementation.
     * @param dataId
     */
    decComplexRef(dataId) {
        return;
    }
    time(f) {
        return notYetImplemented('time');
    }
    read(dataId) {
        return notYetImplemented('read');
    }
    readSync(dataId) {
        return notYetImplemented('readSync');
    }
    numDataIds() {
        return notYetImplemented('numDataIds');
    }
    disposeData(dataId) {
        return notYetImplemented('disposeData');
    }
    write(values, shape, dtype) {
        return notYetImplemented('write');
    }
    move(dataId, values, shape, dtype) {
        return notYetImplemented('move');
    }
    memory() {
        return notYetImplemented('memory');
    }
    /** Returns the highest precision for floats in bits (e.g. 16 or 32) */
    floatPrecision() {
        return notYetImplemented('floatPrecision');
    }
    /** Returns the smallest representable number.  */
    epsilon() {
        return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
    }
    dispose() {
        return notYetImplemented('dispose');
    }
}
function notYetImplemented(kernelName) {
    throw new Error(`'${kernelName}' not yet implemented or not found in the registry. ` +
        `This kernel may not be supported by the tfjs backend you have chosen`);
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
 * Shuffles the array in-place using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1, 2, 3, 4, 5];
 * tf.util.shuffle(a);
 * console.log(a);
 * ```
 *
 * @param array The array to shuffle in-place.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
// tslint:disable-next-line:no-any
function shuffle(array) {
    let counter = array.length;
    let temp = 0;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = (Math.random() * counter) | 0;
        // Decrease counter by 1
        counter--;
        // And swap the last element with it
        temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }
}
/**
 * Shuffles two arrays in-place the same way using Fisher-Yates algorithm.
 *
 * ```js
 * const a = [1,2,3,4,5];
 * const b = [11,22,33,44,55];
 * tf.util.shuffleCombo(a, b);
 * console.log(a, b);
 * ```
 *
 * @param array The first array to shuffle in-place.
 * @param array2 The second array to shuffle in-place with the same permutation
 *     as the first array.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
// tslint:disable-next-line:no-any
function shuffleCombo(array, 
// tslint:disable-next-line:no-any
array2) {
    if (array.length !== array2.length) {
        throw Error(`Array sizes must match to be shuffled together ` +
            `First array length was ${array.length}` +
            `Second array length was ${array2.length}`);
    }
    let counter = array.length;
    let temp, temp2;
    let index = 0;
    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        index = (Math.random() * counter) | 0;
        // Decrease counter by 1
        counter--;
        // And swap the last element of each array with it
        temp = array[counter];
        temp2 = array2[counter];
        array[counter] = array[index];
        array2[counter] = array2[index];
        array[index] = temp;
        array2[index] = temp2;
    }
}
/** Clamps a value to a specified range. */
function clamp(min, x, max) {
    return Math.max(min, Math.min(x, max));
}
function nearestLargerEven(val) {
    return val % 2 === 0 ? val : val + 1;
}
function sum(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum;
}
/**
 * Returns a sample from a uniform [a, b) distribution.
 *
 * @param a The minimum support (inclusive).
 * @param b The maximum support (exclusive).
 * @return A pseudorandom number on the half-open interval [a,b).
 */
function randUniform(a, b) {
    const r = Math.random();
    return (b * r) + (1 - r) * a;
}
/** Returns the squared Euclidean distance between two vectors. */
function distSquared(a, b) {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = Number(a[i]) - Number(b[i]);
        result += diff * diff;
    }
    return result;
}
/**
 * Asserts that the expression is true. Otherwise throws an error with the
 * provided message.
 *
 * ```js
 * const x = 2;
 * tf.util.assert(x === 2, 'x is not 2');
 * ```
 *
 * @param expr The expression to assert (as a boolean).
 * @param msg A function that returns the message to report when throwing an
 *     error. We use a function for performance reasons.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function assert(expr, msg) {
    if (!expr) {
        throw new Error(typeof msg === 'string' ? msg : msg());
    }
}
function assertShapesMatch(shapeA, shapeB, errorMessagePrefix = '') {
    assert(arraysEqual(shapeA, shapeB), () => errorMessagePrefix + ` Shapes ${shapeA} and ${shapeB} must match`);
}
function assertNonNull(a) {
    assert(a != null, () => `The input to the tensor constructor must be a non-null value.`);
}
// NOTE: We explicitly type out what T extends instead of any so that
// util.flatten on a nested array of number doesn't try to infer T as a
// number[][], causing us to explicitly type util.flatten<number>().
/**
 *  Flattens an arbitrarily nested array.
 *
 * ```js
 * const a = [[1, 2], [3, 4], [5, [6, [7]]]];
 * const flat = tf.util.flatten(a);
 * console.log(flat);
 * ```
 *
 *  @param arr The nested array to flatten.
 *  @param result The destination array which holds the elements.
 *  @param skipTypedArray If true, avoids flattening the typed arrays. Defaults
 *      to false.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function flatten(arr, result = [], skipTypedArray = false) {
    if (result == null) {
        result = [];
    }
    if (Array.isArray(arr) || isTypedArray(arr) && !skipTypedArray) {
        for (let i = 0; i < arr.length; ++i) {
            flatten(arr[i], result, skipTypedArray);
        }
    }
    else {
        result.push(arr);
    }
    return result;
}
/**
 * Returns the size (number of elements) of the tensor given its shape.
 *
 * ```js
 * const shape = [3, 4, 2];
 * const size = tf.util.sizeFromShape(shape);
 * console.log(size);
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function sizeFromShape(shape) {
    if (shape.length === 0) {
        // Scalar.
        return 1;
    }
    let size = shape[0];
    for (let i = 1; i < shape.length; i++) {
        size *= shape[i];
    }
    return size;
}
function isScalarShape(shape) {
    return shape.length === 0;
}
function arraysEqual(n1, n2) {
    if (n1 === n2) {
        return true;
    }
    if (n1 == null || n2 == null) {
        return false;
    }
    if (n1.length !== n2.length) {
        return false;
    }
    for (let i = 0; i < n1.length; i++) {
        if (n1[i] !== n2[i]) {
            return false;
        }
    }
    return true;
}
function isInt(a) {
    return a % 1 === 0;
}
function tanh(x) {
    // tslint:disable-next-line:no-any
    if (Math.tanh != null) {
        // tslint:disable-next-line:no-any
        return Math.tanh(x);
    }
    if (x === Infinity) {
        return 1;
    }
    else if (x === -Infinity) {
        return -1;
    }
    else {
        const e2x = Math.exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }
}
function sizeToSquarishShape(size) {
    const width = Math.ceil(Math.sqrt(size));
    return [width, Math.ceil(size / width)];
}
/**
 * Creates a new array with randomized indicies to a given quantity.
 *
 * ```js
 * const randomTen = tf.util.createShuffledIndices(10);
 * console.log(randomTen);
 * ```
 *
 * @param number Quantity of how many shuffled indicies to create.
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function createShuffledIndices(n) {
    const shuffledIndices = new Uint32Array(n);
    for (let i = 0; i < n; ++i) {
        shuffledIndices[i] = i;
    }
    shuffle(shuffledIndices);
    return shuffledIndices;
}
function rightPad(a, size) {
    if (size <= a.length) {
        return a;
    }
    return a + ' '.repeat(size - a.length);
}
function repeatedTry(checkFn, delayFn = (counter) => 0, maxCounter) {
    return new Promise((resolve, reject) => {
        let tryCount = 0;
        const tryFn = () => {
            if (checkFn()) {
                resolve();
                return;
            }
            tryCount++;
            const nextBackoff = delayFn(tryCount);
            if (maxCounter != null && tryCount >= maxCounter) {
                reject();
                return;
            }
            setTimeout(tryFn, nextBackoff);
        };
        tryFn();
    });
}
/**
 * Given the full size of the array and a shape that may contain -1 as the
 * implicit dimension, returns the inferred shape where -1 is replaced.
 * E.g. For shape=[2, -1, 3] and size=24, it will return [2, 4, 3].
 *
 * @param shape The shape, which may contain -1 in some dimension.
 * @param size The full size (number of elements) of the array.
 * @return The inferred shape where -1 is replaced with the inferred size.
 */
function inferFromImplicitShape(shape, size) {
    let shapeProd = 1;
    let implicitIdx = -1;
    for (let i = 0; i < shape.length; ++i) {
        if (shape[i] >= 0) {
            shapeProd *= shape[i];
        }
        else if (shape[i] === -1) {
            if (implicitIdx !== -1) {
                throw Error(`Shapes can only have 1 implicit size. ` +
                    `Found -1 at dim ${implicitIdx} and dim ${i}`);
            }
            implicitIdx = i;
        }
        else if (shape[i] < 0) {
            throw Error(`Shapes can not be < 0. Found ${shape[i]} at dim ${i}`);
        }
    }
    if (implicitIdx === -1) {
        if (size > 0 && size !== shapeProd) {
            throw Error(`Size(${size}) must match the product of shape ${shape}`);
        }
        return shape;
    }
    if (shapeProd === 0) {
        throw Error(`Cannot infer the missing size in [${shape}] when ` +
            `there are 0 elements`);
    }
    if (size % shapeProd !== 0) {
        throw Error(`The implicit shape can't be a fractional number. ` +
            `Got ${size} / ${shapeProd}`);
    }
    const newShape = shape.slice();
    newShape[implicitIdx] = size / shapeProd;
    return newShape;
}
function parseAxisParam(axis, shape) {
    const rank = shape.length;
    // Normalize input
    axis = axis == null ? shape.map((s, i) => i) : [].concat(axis);
    // Check for valid range
    assert(axis.every(ax => ax >= -rank && ax < rank), () => `All values in axis param must be in range [-${rank}, ${rank}) but ` +
        `got axis ${axis}`);
    // Check for only integers
    assert(axis.every(ax => isInt(ax)), () => `All values in axis param must be integers but ` +
        `got axis ${axis}`);
    // Handle negative axis.
    return axis.map(a => a < 0 ? rank + a : a);
}
/** Reduces the shape by removing all dimensions of shape 1. */
function squeezeShape(shape, axis) {
    const newShape = [];
    const keptDims = [];
    const isEmptyArray = axis != null && Array.isArray(axis) && axis.length === 0;
    const axes = (axis == null || isEmptyArray) ?
        null :
        parseAxisParam(axis, shape).sort();
    let j = 0;
    for (let i = 0; i < shape.length; ++i) {
        if (axes != null) {
            if (axes[j] === i && shape[i] !== 1) {
                throw new Error(`Can't squeeze axis ${i} since its dim '${shape[i]}' is not 1`);
            }
            if ((axes[j] == null || axes[j] > i) && shape[i] === 1) {
                newShape.push(shape[i]);
                keptDims.push(i);
            }
            if (axes[j] <= i) {
                j++;
            }
        }
        if (shape[i] !== 1) {
            newShape.push(shape[i]);
            keptDims.push(i);
        }
    }
    return { newShape, keptDims };
}
function getTypedArrayFromDType(dtype, size) {
    let values = null;
    if (dtype == null || dtype === 'float32') {
        values = new Float32Array(size);
    }
    else if (dtype === 'int32') {
        values = new Int32Array(size);
    }
    else if (dtype === 'bool') {
        values = new Uint8Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
    return values;
}
function getArrayFromDType(dtype, size) {
    let values = null;
    if (dtype == null || dtype === 'float32') {
        values = new Float32Array(size);
    }
    else if (dtype === 'int32') {
        values = new Int32Array(size);
    }
    else if (dtype === 'bool') {
        values = new Uint8Array(size);
    }
    else if (dtype === 'string') {
        values = new Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
    return values;
}
function checkConversionForErrors(vals, dtype) {
    for (let i = 0; i < vals.length; i++) {
        const num = vals[i];
        if (isNaN(num) || !isFinite(num)) {
            throw Error(`A tensor of type ${dtype} being uploaded contains ${num}.`);
        }
    }
}
/** Returns true if the dtype is valid. */
function isValidDtype(dtype) {
    return dtype === 'bool' || dtype === 'complex64' || dtype === 'float32' ||
        dtype === 'int32' || dtype === 'string';
}
/**
 * Returns true if the new type can't encode the old type without loss of
 * precision.
 */
function hasEncodingLoss(oldType, newType) {
    if (newType === 'complex64') {
        return false;
    }
    if (newType === 'float32' && oldType !== 'complex64') {
        return false;
    }
    if (newType === 'int32' && oldType !== 'float32' && oldType !== 'complex64') {
        return false;
    }
    if (newType === 'bool' && oldType === 'bool') {
        return false;
    }
    return true;
}
function isTypedArray(a) {
    return a instanceof Float32Array || a instanceof Int32Array ||
        a instanceof Uint8Array;
}
function bytesPerElement(dtype) {
    if (dtype === 'float32' || dtype === 'int32') {
        return 4;
    }
    else if (dtype === 'complex64') {
        return 8;
    }
    else if (dtype === 'bool') {
        return 1;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
/**
 * Returns the approximate number of bytes allocated in the string array - 2
 * bytes per character. Computing the exact bytes for a native string in JS is
 * not possible since it depends on the encoding of the html page that serves
 * the website.
 */
function bytesFromStringArray(arr) {
    if (arr == null) {
        return 0;
    }
    let bytes = 0;
    arr.forEach(x => bytes += x.length);
    return bytes;
}
/** Returns true if the value is a string. */
function isString(value) {
    return typeof value === 'string' || value instanceof String;
}
function isBoolean(value) {
    return typeof value === 'boolean';
}
function isNumber(value) {
    return typeof value === 'number';
}
function inferDtype(values) {
    if (Array.isArray(values)) {
        return inferDtype(values[0]);
    }
    if (values instanceof Float32Array) {
        return 'float32';
    }
    else if (values instanceof Int32Array || values instanceof Uint8Array) {
        return 'int32';
    }
    else if (isNumber(values)) {
        return 'float32';
    }
    else if (isString(values)) {
        return 'string';
    }
    else if (isBoolean(values)) {
        return 'bool';
    }
    return 'float32';
}
function isFunction(f) {
    return !!(f && f.constructor && f.call && f.apply);
}
function nearestDivisor(size, start) {
    for (let i = start; i < size; ++i) {
        if (size % i === 0) {
            return i;
        }
    }
    return size;
}
function computeStrides(shape) {
    const rank = shape.length;
    if (rank < 2) {
        return [];
    }
    // Last dimension has implicit stride of 1, thus having D-1 (instead of D)
    // strides.
    const strides = new Array(rank - 1);
    strides[rank - 2] = shape[rank - 1];
    for (let i = rank - 3; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}
function createNestedArray(offset, shape, a) {
    const ret = new Array();
    if (shape.length === 1) {
        const d = shape[0];
        for (let i = 0; i < d; i++) {
            ret[i] = a[offset + i];
        }
    }
    else {
        const d = shape[0];
        const rest = shape.slice(1);
        const len = rest.reduce((acc, c) => acc * c);
        for (let i = 0; i < d; i++) {
            ret[i] = createNestedArray(offset + i * len, rest, a);
        }
    }
    return ret;
}
// Provide a nested array of TypedArray in given shape.
function toNestedArray(shape, a) {
    if (shape.length === 0) {
        // Scalar type should return a single number.
        return a[0];
    }
    const size = shape.reduce((acc, c) => acc * c);
    if (size === 0) {
        // A tensor with shape zero should be turned into empty list.
        return [];
    }
    if (size !== a.length) {
        throw new Error(`[${shape}] does not match the input size ${a.length}.`);
    }
    return createNestedArray(0, shape, a);
}
function makeOnesTypedArray(size, dtype) {
    const array = makeZerosTypedArray(size, dtype);
    for (let i = 0; i < array.length; i++) {
        array[i] = 1;
    }
    return array;
}
function makeZerosTypedArray(size, dtype) {
    if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
        return new Float32Array(size);
    }
    else if (dtype === 'int32') {
        return new Int32Array(size);
    }
    else if (dtype === 'bool') {
        return new Uint8Array(size);
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
/**
 * Make nested `TypedArray` filled with zeros.
 * @param shape The shape information for the nested array.
 * @param dtype dtype of the array element.
 */
function makeZerosNestedTypedArray(shape, dtype) {
    const size = shape.reduce((prev, curr) => prev * curr, 1);
    if (dtype == null || dtype === 'float32') {
        return toNestedArray(shape, new Float32Array(size));
    }
    else if (dtype === 'int32') {
        return toNestedArray(shape, new Int32Array(size));
    }
    else if (dtype === 'bool') {
        return toNestedArray(shape, new Uint8Array(size));
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
function assertNonNegativeIntegerDimensions(shape) {
    shape.forEach(dimSize => {
        assert(Number.isInteger(dimSize) && dimSize >= 0, () => `Tensor must have a shape comprised of positive integers but got ` +
            `shape [${shape}].`);
    });
}
/**
 * Computes flat index for a given location (multidimentionsal index) in a
 * Tensor/multidimensional array.
 *
 * @param locs Location in the tensor.
 * @param rank Rank of the tensor.
 * @param strides Tensor strides.
 */
function locToIndex(locs, rank, strides) {
    if (rank === 0) {
        return 0;
    }
    else if (rank === 1) {
        return locs[0];
    }
    let index = locs[locs.length - 1];
    for (let i = 0; i < locs.length - 1; ++i) {
        index += strides[i] * locs[i];
    }
    return index;
}
/**
 * Computes the location (multidimensional index) in a tensor/multidimentional
 * array for a given flat index.
 *
 * @param index Index in flat array.
 * @param rank Rank of tensor.
 * @param strides Strides of tensor.
 */
function indexToLoc(index, rank, strides) {
    if (rank === 0) {
        return [];
    }
    else if (rank === 1) {
        return [index];
    }
    const locs = new Array(rank);
    for (let i = 0; i < locs.length - 1; ++i) {
        locs[i] = Math.floor(index / strides[i]);
        index -= locs[i] * strides[i];
    }
    locs[locs.length - 1] = index;
    return locs;
}
/**
 * This method asserts whether an object is a Promise instance.
 * @param object
 */
// tslint:disable-next-line: no-any
function isPromise(object) {
    //  We chose to not use 'obj instanceOf Promise' for two reasons:
    //  1. It only reliably works for es6 Promise, not other Promise
    //  implementations.
    //  2. It doesn't work with framework that uses zone.js. zone.js monkey patch
    //  the async calls, so it is possible the obj (patched) is comparing to a
    //  pre-patched Promise.
    return object && object.then && typeof object.then === 'function';
}

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
// Expects flags from URL in the format ?tfjsflags=FLAG1:1,FLAG2:true.
const TENSORFLOWJS_FLAGS_PREFIX = 'tfjsflags';
/**
 * The environment contains evaluated flags as well as the registered platform.
 * This is always used as a global singleton and can be retrieved with
 * `tf.env()`.
 *
 * @doc {heading: 'Environment'}
 */
class Environment {
    // tslint:disable-next-line: no-any
    constructor(global) {
        this.global = global;
        this.flags = {};
        this.flagRegistry = {};
        this.urlFlags = {};
        this.populateURLFlags();
    }
    setPlatform(platformName, platform) {
        if (this.platform != null) {
            console.warn(`Platform ${this.platformName} has already been set. ` +
                `Overwriting the platform with ${platform}.`);
        }
        this.platformName = platformName;
        this.platform = platform;
    }
    registerFlag(flagName, evaluationFn, setHook) {
        this.flagRegistry[flagName] = { evaluationFn, setHook };
        // Override the flag value from the URL. This has to happen here because the
        // environment is initialized before flags get registered.
        if (this.urlFlags[flagName] != null) {
            const flagValue = this.urlFlags[flagName];
            console.warn(`Setting feature override from URL ${flagName}: ${flagValue}.`);
            this.set(flagName, flagValue);
        }
    }
    async getAsync(flagName) {
        if (flagName in this.flags) {
            return this.flags[flagName];
        }
        this.flags[flagName] = await this.evaluateFlag(flagName);
        return this.flags[flagName];
    }
    get(flagName) {
        if (flagName in this.flags) {
            return this.flags[flagName];
        }
        const flagValue = this.evaluateFlag(flagName);
        if (isPromise(flagValue)) {
            throw new Error(`Flag ${flagName} cannot be synchronously evaluated. ` +
                `Please use getAsync() instead.`);
        }
        this.flags[flagName] = flagValue;
        return this.flags[flagName];
    }
    getNumber(flagName) {
        return this.get(flagName);
    }
    getBool(flagName) {
        return this.get(flagName);
    }
    getFlags() {
        return this.flags;
    }
    // For backwards compatibility.
    get features() {
        return this.flags;
    }
    set(flagName, value) {
        if (this.flagRegistry[flagName] == null) {
            throw new Error(`Cannot set flag ${flagName} as it has not been registered.`);
        }
        this.flags[flagName] = value;
        if (this.flagRegistry[flagName].setHook != null) {
            this.flagRegistry[flagName].setHook(value);
        }
    }
    evaluateFlag(flagName) {
        if (this.flagRegistry[flagName] == null) {
            throw new Error(`Cannot evaluate flag '${flagName}': no evaluation function found.`);
        }
        return this.flagRegistry[flagName].evaluationFn();
    }
    setFlags(flags) {
        this.flags = Object.assign({}, flags);
    }
    reset() {
        this.flags = {};
        this.urlFlags = {};
        this.populateURLFlags();
    }
    populateURLFlags() {
        if (typeof this.global === 'undefined' ||
            typeof this.global.location === 'undefined' ||
            typeof this.global.location.search === 'undefined') {
            return;
        }
        const urlParams = getQueryParams(this.global.location.search);
        if (TENSORFLOWJS_FLAGS_PREFIX in urlParams) {
            const keyValues = urlParams[TENSORFLOWJS_FLAGS_PREFIX].split(',');
            keyValues.forEach(keyValue => {
                const [key, value] = keyValue.split(':');
                this.urlFlags[key] = parseValue(key, value);
            });
        }
    }
}
function getQueryParams(queryString) {
    const params = {};
    queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, (s, ...t) => {
        decodeParam(params, t[0], t[1]);
        return t.join('=');
    });
    return params;
}
function decodeParam(params, name, value) {
    params[decodeURIComponent(name)] = decodeURIComponent(value || '');
}
function parseValue(flagName, value) {
    value = value.toLowerCase();
    if (value === 'true' || value === 'false') {
        return value === 'true';
    }
    else if (`${+value}` === value) {
        return +value;
    }
    throw new Error(`Could not parse value flag value ${value} for flag ${flagName}.`);
}
/**
 * Returns the current environment (a global singleton).
 *
 * The environment object contains the evaluated feature values as well as the
 * active platform.
 *
 * @doc {heading: 'Environment'}
 */
function env() {
    return ENV;
}
let ENV = null;
function setEnvironmentGlobal(environment) {
    ENV = environment;
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
// Note that the identifier globalNameSpace is scoped to this module, but will
// always resolve to the same global object regardless of how the module is
// resolved.
// tslint:disable-next-line:no-any
let globalNameSpace;
// tslint:disable-next-line:no-any
function getGlobalNamespace() {
    if (globalNameSpace == null) {
        // tslint:disable-next-line:no-any
        let ns;
        if (typeof (window) !== 'undefined') {
            ns = window;
        }
        else if (typeof (global) !== 'undefined') {
            ns = global;
        }
        else if (typeof (process) !== 'undefined') {
            ns = process;
        }
        else if (typeof (self) !== 'undefined') {
            ns = self;
        }
        else {
            throw new Error('Could not find a global object');
        }
        globalNameSpace = ns;
    }
    return globalNameSpace;
}
// tslint:disable-next-line:no-any
function getGlobalMap() {
    const ns = getGlobalNamespace();
    if (ns._tfGlobals == null) {
        ns._tfGlobals = new Map();
    }
    return ns._tfGlobals;
}
/**
 * Returns a globally accessible 'singleton' object.
 *
 * @param key the name of the object
 * @param init a function to initialize to initialize this object
 *             the first time it is fetched.
 */
function getGlobal(key, init) {
    const globalMap = getGlobalMap();
    if (globalMap.has(key)) {
        return globalMap.get(key);
    }
    else {
        const singleton = init();
        globalMap.set(key, singleton);
        return globalMap.get(key);
    }
}

const Abs = 'Abs';
const Acos = 'Acos';
const Acosh = 'Acosh';
const Add = 'Add';
const AddN = 'AddN';
const All = 'All';
const Any = 'Any';
const ArgMax = 'ArgMax';
const ArgMin = 'ArgMin';
const Asin = 'Asin';
const Asinh = 'Asinh';
const Atan = 'Atan';
const Atanh = 'Atanh';
const Atan2 = 'Atan2';
const AvgPool = 'AvgPool';
const AvgPoolGrad = 'AvgPoolGrad';
const AvgPool3D = 'AvgPool3D';
const AvgPool3DGrad = 'AvgPool3DGrad';
const BatchMatMul = 'BatchMatMul';
const BatchToSpaceND = 'BatchToSpaceND';
const Bincount = 'Bincount';
const BroadcastTo = 'BroadcastTo';
const Cast = 'Cast';
const Ceil = 'Ceil';
const ClipByValue = 'ClipByValue';
const Complex = 'Complex';
const ComplexAbs = 'ComplexAbs';
const Concat = 'Concat';
const Conv2D = 'Conv2D';
const Conv2DBackpropFilter = 'Conv2DBackpropFilter';
const Conv2DBackpropInput = 'Conv2DBackpropInput';
const Conv3D = 'Conv3D';
const Conv3DBackpropFilterV2 = 'Conv3DBackpropFilterV2';
const Conv3DBackpropInputV2 = 'Conv3DBackpropInputV2';
const Cos = 'Cos';
const Cosh = 'Cosh';
const Cumsum = 'Cumsum';
const CropAndResize = 'CropAndResize';
const DenseBincount = 'DenseBincount';
const DepthToSpace = 'DepthToSpace';
const DepthwiseConv2dNative = 'DepthwiseConv2dNative';
const DepthwiseConv2dNativeBackpropFilter = 'DepthwiseConv2dNativeBackpropFilter';
const DepthwiseConv2dNativeBackpropInput = 'DepthwiseConv2dNativeBackpropInput';
const Diag = 'Diag';
const Dilation2D = 'Dilation2D';
const Dilation2DBackpropInput = 'Dilation2DBackpropInput';
const Dilation2DBackpropFilter = 'Dilation2DBackpropFilter';
const RealDiv = 'RealDiv';
const Elu = 'Elu';
const EluGrad = 'EluGrad';
const Erf = 'Erf';
const Equal = 'Equal';
const Exp = 'Exp';
const ExpandDims = 'ExpandDims';
const Expm1 = 'Expm1';
const FFT = 'FFT';
const Fill = 'Fill';
const FlipLeftRight = 'FlipLeftRight';
const Floor = 'Floor';
const FloorDiv = 'FloorDiv';
const FusedBatchNorm = 'FusedBatchNorm';
const GatherV2 = 'GatherV2';
const GatherNd = 'GatherNd';
const Greater = 'Greater';
const GreaterEqual = 'GreaterEqual';
const Identity = 'Identity';
const IFFT = 'IFFT';
const Imag = 'Imag';
const IsFinite = 'IsFinite';
const IsInf = 'IsInf';
const IsNan = 'IsNan';
const LeakyRelu = 'LeakyRelu';
const Less = 'Less';
const LessEqual = 'LessEqual';
const LinSpace = 'LinSpace';
const Log = 'Log';
const Log1p = 'Log1p';
const LogicalAnd = 'LogicalAnd';
const LogicalNot = 'LogicalNot';
const LogicalOr = 'LogicalOr';
const LogSoftmax = 'LogSoftmax';
const LRN = 'LRN';
const LRNGrad = 'LRNGrad';
const Max = 'Max';
const Maximum = 'Maximum';
const MaxPool = 'MaxPool';
const MaxPoolGrad = 'MaxPoolGrad';
const MaxPool3D = 'MaxPool3D';
const MaxPool3DGrad = 'MaxPool3DGrad';
const MaxPoolWithArgmax = 'MaxPoolWithArgmax';
const Mean = 'Mean';
const Min = 'Min';
const Minimum = 'Minimum';
const MirrorPad = 'MirrorPad';
const Mod = 'Mod';
const Multinomial = 'Multinomial';
const Multiply = 'Multiply';
const Neg = 'Neg';
const NotEqual = 'NotEqual';
const NonMaxSuppressionV3 = 'NonMaxSuppressionV3';
const NonMaxSuppressionV4 = 'NonMaxSuppressionV4';
const NonMaxSuppressionV5 = 'NonMaxSuppressionV5';
const OnesLike = 'OnesLike';
const OneHot = 'OneHot';
const Pack = 'Pack';
const PadV2 = 'PadV2';
const Pool = 'Pool';
const Pow = 'Pow';
const Prelu = 'Prelu';
const Prod = 'Prod';
const Range = 'Range';
const Real = 'Real';
const Reciprocal = 'Reciprocal';
const Relu = 'Relu';
const Reshape = 'Reshape';
const ResizeNearestNeighbor = 'ResizeNearestNeighbor';
const ResizeNearestNeighborGrad = 'ResizeNearestNeighborGrad';
const ResizeBilinear = 'ResizeBilinear';
const ResizeBilinearGrad = 'ResizeBilinearGrad';
const Relu6 = 'Relu6';
const Reverse = 'Reverse';
const Round = 'Round';
const Rsqrt = 'Rsqrt';
const ScatterNd = 'ScatterNd';
const Select = 'Select';
const Selu = 'Selu';
const Slice = 'Slice';
const Sin = 'Sin';
const Sinh = 'Sinh';
const Sign = 'Sign';
const Sigmoid = 'Sigmoid';
const Softplus = 'Softplus';
const Sqrt = 'Sqrt';
const Sum = 'Sum';
const SpaceToBatchND = 'SpaceToBatchND';
const SplitV = 'SplitV';
const Softmax = 'Softmax';
const SquaredDifference = 'SquaredDifference';
const Square = 'Square';
const Sub = 'Sub';
const SparseToDense = 'SparseToDense';
const StridedSlice = 'StridedSlice';
const Tan = 'Tan';
const Tanh = 'Tanh';
const Tile = 'Tile';
const TopK = 'TopK';
const Transpose = 'Transpose';
const Unique = 'Unique';
const Unpack = 'Unpack';
const UnsortedSegmentSum = 'UnsortedSegmentSum';
const ZerosLike = 'ZerosLike';
/**
 * TensorFlow.js-only kernels
 */
const Step = 'Step';
const FromPixels = 'FromPixels';
const RotateWithOffset = 'RotateWithOffset';
const _FusedMatMul = '_FusedMatMul';
const FusedConv2D = 'FusedConv2D';
const FusedDepthwiseConv2D = 'FusedDepthwiseConv2D';

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
const kernelRegistry = getGlobal('kernelRegistry', () => new Map());
const gradRegistry = getGlobal('gradRegistry', () => new Map());
/**
 * Returns the kernel function (code) associated with the provided names.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 */
function getKernel(kernelName, backendName) {
    const key = makeKey(kernelName, backendName);
    return kernelRegistry.get(key);
}
/**
 * Returns the registered gradient info associated with the provided kernel.
 * @param kernelName The official TF kernel name.
 */
function getGradient(kernelName) {
    return gradRegistry.get(kernelName);
}
function getKernelsForBackend(backendName) {
    const it = kernelRegistry.entries();
    const result = [];
    while (true) {
        const { done, value } = it.next();
        if (done) {
            break;
        }
        const [key, config] = value;
        const [backend,] = key.split('_');
        if (backend === backendName) {
            result.push(config);
        }
    }
    return result;
}
/**
 * Registers the function (forward pass) for the kernel in a global registry.
 *
 * @param config A config object with the following properties:
 * - `kernelName` The official name of the kernel.
 * - `backendName` The official name of the backend.
 * - `kernelFunc` The function to run during the forward pass of the kernel.
 * - `setupFunc` Optional. Gets called once, after the backend initializes.
 * - `disposeFunc` Optional. Gets called once, right before the backend is
 * disposed.
 */
function registerKernel(config) {
    const { kernelName, backendName } = config;
    const key = makeKey(kernelName, backendName);
    if (kernelRegistry.has(key)) {
        console.warn(`The kernel '${kernelName}' for backend ` +
            `'${backendName}' is already registered`);
    }
    kernelRegistry.set(key, config);
}
/**
 * Registers a gradient function for a given kernel in the global registry,
 * to be used during the back-propagation of that kernel.
 *
 * @param config An object with the following properties:
 * - `kernelName` The name of the kernel that the gradient function is for.
 * - `gradFunc` The function to run during back-propagation.
 */
function registerGradient(config) {
    const { kernelName } = config;
    if (gradRegistry.has(kernelName)) {
        // TODO (yassogba) after 3.0 assess whether we need to keep this gated
        // to debug mode.
        if (env().getBool('DEBUG')) {
            console.warn(`Overriding the gradient for '${kernelName}'`);
        }
    }
    gradRegistry.set(kernelName, config);
}
/**
 * Removes the kernel function from the registry.
 *
 * @param kernelName The official name of the kernel.
 * @param backendName The official name of the backend.
 *
 */
function unregisterKernel(kernelName, backendName) {
    const key = makeKey(kernelName, backendName);
    if (!kernelRegistry.has(key)) {
        throw new Error(`The kernel '${kernelName}' for backend ` +
            `'${backendName}' is not registered`);
    }
    kernelRegistry.delete(key);
}
/** Removes the registered gradient from the global registry. */
function unregisterGradient(kernelName) {
    if (!gradRegistry.has(kernelName)) {
        throw new Error(`The gradient '${kernelName}' for backend is not registered`);
    }
    gradRegistry.delete(kernelName);
}
/**
 * Finds kernels that have already been registered to a backend and re-registers
 * them for a new backend. Useful for registering custom backends.
 * @param registeredBackendName Already registered backend.
 * @param newBackendName New backend.
 */
function copyRegisteredKernels(registeredBackendName, newBackendName) {
    const kernels = getKernelsForBackend(registeredBackendName);
    kernels.forEach(kernelConfig => {
        const newKernelConfig = Object.assign({}, kernelConfig, { backendName: newBackendName });
        registerKernel(newKernelConfig);
    });
}
function makeKey(kernelName, backendName) {
    return `${backendName}_${kernelName}`;
}

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
/**
 * Create typed array for scalar value. Used for storing in `DataStorage`.
 */
function createScalarValue(value, dtype) {
    if (dtype === 'string') {
        return encodeString(value);
    }
    return toTypedArray([value], dtype);
}
function noConversionNeeded(a, dtype) {
    return (a instanceof Float32Array && dtype === 'float32') ||
        (a instanceof Int32Array && dtype === 'int32') ||
        (a instanceof Uint8Array && dtype === 'bool');
}
function toTypedArray(a, dtype) {
    if (dtype === 'string') {
        throw new Error('Cannot convert a string[] to a TypedArray');
    }
    if (Array.isArray(a)) {
        a = flatten(a);
    }
    if (env().getBool('DEBUG')) {
        checkConversionForErrors(a, dtype);
    }
    if (noConversionNeeded(a, dtype)) {
        return a;
    }
    if (dtype == null || dtype === 'float32' || dtype === 'complex64') {
        return new Float32Array(a);
    }
    else if (dtype === 'int32') {
        return new Int32Array(a);
    }
    else if (dtype === 'bool') {
        const bool = new Uint8Array(a.length);
        for (let i = 0; i < bool.length; ++i) {
            if (Math.round(a[i]) !== 0) {
                bool[i] = 1;
            }
        }
        return bool;
    }
    else {
        throw new Error(`Unknown data type ${dtype}`);
    }
}
/**
 * Returns the current high-resolution time in milliseconds relative to an
 * arbitrary time in the past. It works across different platforms (node.js,
 * browsers).
 *
 * ```js
 * console.log(tf.util.now());
 * ```
 *
 * @doc {heading: 'Util', namespace: 'util'}
 */
function now() {
    return env().platform.now();
}
/**
 * Returns a platform-specific implementation of
 * [`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
 *
 * If `fetch` is defined on the global object (`window`, `process`, etc.),
 * `tf.util.fetch` returns that function.
 *
 * If not, `tf.util.fetch` returns a platform-specific solution.
 *
 * ```js
 * const resource = await tf.util.fetch('https://unpkg.com/@tensorflow/tfjs');
 * // handle response
 * ```
 *
 * @doc {heading: 'Util'}
 */
function fetch$1(path, requestInits) {
    return env().platform.fetch(path, requestInits);
}
/**
 * Encodes the provided string into bytes using the provided encoding scheme.
 *
 * @param s The string to encode.
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
function encodeString(s, encoding = 'utf-8') {
    encoding = encoding || 'utf-8';
    return env().platform.encode(s, encoding);
}
/**
 * Decodes the provided bytes into a string using the provided encoding scheme.
 * @param bytes The bytes to decode.
 *
 * @param encoding The encoding scheme. Defaults to utf-8.
 *
 * @doc {heading: 'Util'}
 */
function decodeString(bytes, encoding = 'utf-8') {
    encoding = encoding || 'utf-8';
    return env().platform.decode(bytes, encoding);
}

var util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    createScalarValue: createScalarValue,
    toTypedArray: toTypedArray,
    now: now,
    fetch: fetch$1,
    encodeString: encodeString,
    decodeString: decodeString,
    shuffle: shuffle,
    shuffleCombo: shuffleCombo,
    clamp: clamp,
    nearestLargerEven: nearestLargerEven,
    sum: sum,
    randUniform: randUniform,
    distSquared: distSquared,
    assert: assert,
    assertShapesMatch: assertShapesMatch,
    assertNonNull: assertNonNull,
    flatten: flatten,
    sizeFromShape: sizeFromShape,
    isScalarShape: isScalarShape,
    arraysEqual: arraysEqual,
    isInt: isInt,
    tanh: tanh,
    sizeToSquarishShape: sizeToSquarishShape,
    createShuffledIndices: createShuffledIndices,
    rightPad: rightPad,
    repeatedTry: repeatedTry,
    inferFromImplicitShape: inferFromImplicitShape,
    parseAxisParam: parseAxisParam,
    squeezeShape: squeezeShape,
    getTypedArrayFromDType: getTypedArrayFromDType,
    getArrayFromDType: getArrayFromDType,
    checkConversionForErrors: checkConversionForErrors,
    isValidDtype: isValidDtype,
    hasEncodingLoss: hasEncodingLoss,
    isTypedArray: isTypedArray,
    bytesPerElement: bytesPerElement,
    bytesFromStringArray: bytesFromStringArray,
    isString: isString,
    isBoolean: isBoolean,
    isNumber: isNumber,
    inferDtype: inferDtype,
    isFunction: isFunction,
    nearestDivisor: nearestDivisor,
    computeStrides: computeStrides,
    toNestedArray: toNestedArray,
    makeOnesTypedArray: makeOnesTypedArray,
    makeZerosTypedArray: makeZerosTypedArray,
    makeZerosNestedTypedArray: makeZerosNestedTypedArray,
    assertNonNegativeIntegerDimensions: assertNonNegativeIntegerDimensions,
    locToIndex: locToIndex,
    indexToLoc: indexToLoc,
    isPromise: isPromise
});

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
class Profiler {
    constructor(backendTimer, logger) {
        this.backendTimer = backendTimer;
        this.logger = logger;
        if (logger == null) {
            this.logger = new Logger();
        }
    }
    profileKernel(kernelName, inputs, f) {
        let outputs;
        const holdResultWrapperFn = () => {
            outputs = f();
        };
        const timer = this.backendTimer.time(holdResultWrapperFn);
        if (env().getBool('CHECK_COMPUTATION_FOR_ERRORS')) {
            for (let i = 0; i < outputs.length; i++) {
                const output = outputs[i];
                // Dangling promise here because we don't want to propagate up
                // asynchronicity.
                output.data().then(tensorVals => {
                    checkComputationForErrors(tensorVals, output.dtype, kernelName);
                });
            }
        }
        const kernelProfile = {
            kernelName,
            outputs,
            inputs,
            timeMs: timer.then(timing => timing.kernelMs),
            extraInfo: timer.then(timing => timing.getExtraProfileInfo != null ?
                timing.getExtraProfileInfo() :
                '')
        };
        return kernelProfile;
    }
    logKernelProfile(kernelProfile) {
        const { kernelName, outputs, timeMs, inputs, extraInfo } = kernelProfile;
        outputs.forEach(result => {
            Promise.all([result.data(), timeMs, extraInfo]).then(valueContainer => {
                this.logger.logKernelProfile(kernelName, result, valueContainer[0], valueContainer[1], inputs, valueContainer[2]);
            });
        });
    }
}
function checkComputationForErrors(vals, dtype, kernelName) {
    if (dtype !== 'float32') {
        // Only floating point computations will generate NaN values
        return false;
    }
    for (let i = 0; i < vals.length; i++) {
        const num = vals[i];
        if (isNaN(num) || !isFinite(num)) {
            // Throwing custom exception so behavior is testable.
            console.warn(`Found ${num} in the result of '${kernelName}'`);
            return true;
        }
    }
    return false;
}
class Logger {
    logKernelProfile(name, result, vals, timeMs, inputs, extraInfo) {
        const time = typeof timeMs === 'number' ? rightPad(`${timeMs}ms`, 9) :
            timeMs['error'];
        const paddedName = rightPad(name, 25);
        const rank = result.rank;
        const size = result.size;
        const shape = rightPad(result.shape.toString(), 14);
        let inputShapesDescription = '';
        for (const name in inputs) {
            const input = inputs[name];
            if (input != null) {
                // The input might be a non-tensor (e.g HTMLImageElement), in which case
                // we claim the output shape as input shape.
                const inputShape = input.shape || result.shape;
                const inputRank = inputShape.length;
                inputShapesDescription +=
                    `${name}: ${inputRank}D ${inputRank > 0 ? inputShape : ''} `;
            }
        }
        console.log(`%c${paddedName}\t%c${time}\t%c${rank}D ${shape}\t%c${size}\t%c${inputShapesDescription}\t%c${extraInfo}`, 'font-weight:bold', 'color:red', 'color:blue', 'color: orange', 'color: green', 'color: steelblue');
    }
}

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
/**
 * Computes a list of TapeNodes that connect x to y, filtering everything else
 * out and preserving the order of the original tape elements.
 *
 * @param tape The tape elements to filter.
 * @param xs The input Tensors.
 * @param y The output Tensor.
 */
function getFilteredNodesXToY(tape, xs, y) {
    // Forward pass to compute all the nodes and Tensors that are transitively a
    // function of x.
    const tensorsFromX = {};
    const nodesFromX = {};
    for (let i = 0; i < xs.length; i++) {
        tensorsFromX[xs[i].id] = true;
    }
    for (let i = 0; i < tape.length; i++) {
        const node = tape[i];
        const nodeInputs = node.inputs;
        for (const inputName in nodeInputs) {
            const input = nodeInputs[inputName];
            let anyInputFromX = false;
            for (let j = 0; j < xs.length; j++) {
                if (tensorsFromX[input.id]) {
                    node.outputs.forEach(output => tensorsFromX[output.id] = true);
                    anyInputFromX = true;
                    nodesFromX[node.id] = true;
                    break;
                }
            }
            if (anyInputFromX) {
                break;
            }
        }
    }
    // Backward pass to find all of the nodes and Tensors that lead to y.
    const tensorsLeadToY = {};
    tensorsLeadToY[y.id] = true;
    const nodesToY = {};
    for (let i = tape.length - 1; i >= 0; i--) {
        const node = tape[i];
        const nodeInputs = node.inputs;
        // If any of the outputs lead to y, mark all of the inputs as leading to y.
        for (let j = 0; j < node.outputs.length; j++) {
            if (tensorsLeadToY[node.outputs[j].id]) {
                for (const inputName in nodeInputs) {
                    tensorsLeadToY[nodeInputs[inputName].id] = true;
                    nodesToY[node.id] = true;
                }
                break;
            }
        }
    }
    // Return the paths that come from x and lead to y.
    const filteredTape = [];
    for (let i = 0; i < tape.length; i++) {
        const node = tape[i];
        if (nodesFromX[node.id] && nodesToY[node.id]) {
            // Prune the inputs from the node that aren't a function of x.
            const prunedInputs = {};
            for (const inputName in node.inputs) {
                const nodeInput = node.inputs[inputName];
                if (tensorsFromX[nodeInput.id]) {
                    prunedInputs[inputName] = nodeInput;
                }
            }
            // Copy the node and overwrite inputsAndArgs to the pruned version.
            const prunedNode = Object.assign({}, node);
            prunedNode.inputs = prunedInputs;
            prunedNode.outputs = node.outputs;
            filteredTape.push(prunedNode);
        }
    }
    return filteredTape;
}
/**
 * Backpropagate gradients through the filtered TapeNodes.
 *
 * @param tensorAccumulatedGradientMap A map of Tensor to its gradient. This map
 * is mutated by this method.
 * @param filteredTape The filtered TapeNodes to backprop through.
 */
function backpropagateGradients(tensorAccumulatedGradientMap, filteredTape, tidy, add) {
    // Walk the tape backward and keep a map of Tensor to its gradient.
    for (let i = filteredTape.length - 1; i >= 0; i--) {
        const node = filteredTape[i];
        const dys = [];
        node.outputs.forEach(o => {
            const gradTensor = tensorAccumulatedGradientMap[o.id];
            if (gradTensor != null) {
                dys.push(gradTensor);
            }
            else {
                // This particular output is not in the back-propagation subgraph, so it
                // does not affect the final output, thus we put null for its dy.
                dys.push(null);
            }
        });
        if (node.gradient == null) {
            throw new Error(`Cannot compute gradient: gradient function not found ` +
                `for ${node.kernelName}.`);
        }
        // Backprop dy through this node and accumulate gradients over the inputs.
        const inputGradients = node.gradient(dys);
        for (const inputName in node.inputs) {
            if (!(inputName in inputGradients)) {
                throw new Error(`Cannot backprop through input ${inputName}. ` +
                    `Available gradients found: ${Object.keys(inputGradients)}.`);
            }
            // Call the gradient function.
            const dx = tidy(() => inputGradients[inputName]());
            if (dx.dtype !== 'float32') {
                throw new Error(`Error in gradient for op ${node.kernelName}. The gradient of input ` +
                    `${inputName} must have 'float32' dtype, but has '${dx.dtype}'`);
            }
            const x = node.inputs[inputName];
            if (!arraysEqual(dx.shape, x.shape)) {
                throw new Error(`Error in gradient for op ${node.kernelName}. The gradient of input ` +
                    `'${inputName}' has shape '${dx.shape}', which does not match ` +
                    `the shape of the input '${x.shape}'`);
            }
            if (tensorAccumulatedGradientMap[x.id] == null) {
                tensorAccumulatedGradientMap[x.id] = dx;
            }
            else {
                const curGradient = tensorAccumulatedGradientMap[x.id];
                tensorAccumulatedGradientMap[x.id] = add(curGradient, dx);
                curGradient.dispose();
            }
        }
    }
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
// Maximum number of values before we decide to show ellipsis.
const FORMAT_LIMIT_NUM_VALS = 20;
// Number of first and last values to show when displaying a, b,...,y, z.
const FORMAT_NUM_FIRST_LAST_VALS = 3;
// Number of significant digits to show.
const FORMAT_NUM_SIG_DIGITS = 7;
function tensorToString(vals, shape, dtype, verbose) {
    const strides = computeStrides(shape);
    const padPerCol = computeMaxSizePerColumn(vals, shape, dtype, strides);
    const rank = shape.length;
    const valsLines = subTensorToString(vals, shape, dtype, strides, padPerCol);
    const lines = ['Tensor'];
    if (verbose) {
        lines.push(`  dtype: ${dtype}`);
        lines.push(`  rank: ${rank}`);
        lines.push(`  shape: [${shape}]`);
        lines.push(`  values:`);
    }
    lines.push(valsLines.map(l => '    ' + l).join('\n'));
    return lines.join('\n');
}
function computeMaxSizePerColumn(vals, shape, dtype, strides) {
    const n = sizeFromShape(shape);
    const numCols = strides[strides.length - 1];
    const padPerCol = new Array(numCols).fill(0);
    const rank = shape.length;
    const valuesOrTuples = dtype === 'complex64' ? createComplexTuples(vals) : vals;
    if (rank > 1) {
        for (let row = 0; row < n / numCols; row++) {
            const offset = row * numCols;
            for (let j = 0; j < numCols; j++) {
                padPerCol[j] = Math.max(padPerCol[j], valToString(valuesOrTuples[offset + j], 0, dtype).length);
            }
        }
    }
    return padPerCol;
}
function valToString(val, pad, dtype) {
    let valStr;
    if (Array.isArray(val)) {
        valStr = `${parseFloat(val[0].toFixed(FORMAT_NUM_SIG_DIGITS))} + ` +
            `${parseFloat(val[1].toFixed(FORMAT_NUM_SIG_DIGITS))}j`;
    }
    else if (isString(val)) {
        valStr = `'${val}'`;
    }
    else if (dtype === 'bool') {
        valStr = boolNumToString(val);
    }
    else {
        valStr = parseFloat(val.toFixed(FORMAT_NUM_SIG_DIGITS)).toString();
    }
    return rightPad(valStr, pad);
}
function boolNumToString(v) {
    return v === 0 ? 'false' : 'true';
}
function subTensorToString(vals, shape, dtype, strides, padPerCol, isLast = true) {
    const storagePerElement = dtype === 'complex64' ? 2 : 1;
    const size = shape[0];
    const rank = shape.length;
    if (rank === 0) {
        if (dtype === 'complex64') {
            const complexTuple = createComplexTuples(vals);
            return [valToString(complexTuple[0], 0, dtype)];
        }
        if (dtype === 'bool') {
            return [boolNumToString(vals[0])];
        }
        return [vals[0].toString()];
    }
    if (rank === 1) {
        if (size > FORMAT_LIMIT_NUM_VALS) {
            const firstValsSize = FORMAT_NUM_FIRST_LAST_VALS * storagePerElement;
            let firstVals = Array.from(vals.slice(0, firstValsSize));
            let lastVals = Array.from(vals.slice((size - FORMAT_NUM_FIRST_LAST_VALS) * storagePerElement, size * storagePerElement));
            if (dtype === 'complex64') {
                firstVals = createComplexTuples(firstVals);
                lastVals = createComplexTuples(lastVals);
            }
            return [
                '[' +
                    firstVals.map((x, i) => valToString(x, padPerCol[i], dtype))
                        .join(', ') +
                    ', ..., ' +
                    lastVals
                        .map((x, i) => valToString(x, padPerCol[size - FORMAT_NUM_FIRST_LAST_VALS + i], dtype))
                        .join(', ') +
                    ']'
            ];
        }
        const displayVals = dtype === 'complex64' ? createComplexTuples(vals) :
            Array.from(vals);
        return [
            '[' +
                displayVals.map((x, i) => valToString(x, padPerCol[i], dtype))
                    .join(', ') +
                ']'
        ];
    }
    // The array is rank 2 or more.
    const subshape = shape.slice(1);
    const substrides = strides.slice(1);
    const stride = strides[0] * storagePerElement;
    const lines = [];
    if (size > FORMAT_LIMIT_NUM_VALS) {
        for (let i = 0; i < FORMAT_NUM_FIRST_LAST_VALS; i++) {
            const start = i * stride;
            const end = start + stride;
            lines.push(...subTensorToString(vals.slice(start, end), subshape, dtype, substrides, padPerCol, false /* isLast */));
        }
        lines.push('...');
        for (let i = size - FORMAT_NUM_FIRST_LAST_VALS; i < size; i++) {
            const start = i * stride;
            const end = start + stride;
            lines.push(...subTensorToString(vals.slice(start, end), subshape, dtype, substrides, padPerCol, i === size - 1 /* isLast */));
        }
    }
    else {
        for (let i = 0; i < size; i++) {
            const start = i * stride;
            const end = start + stride;
            lines.push(...subTensorToString(vals.slice(start, end), subshape, dtype, substrides, padPerCol, i === size - 1 /* isLast */));
        }
    }
    const sep = rank === 2 ? ',' : '';
    lines[0] = '[' + lines[0] + sep;
    for (let i = 1; i < lines.length - 1; i++) {
        lines[i] = ' ' + lines[i] + sep;
    }
    let newLineSep = ',\n';
    for (let i = 2; i < rank; i++) {
        newLineSep += '\n';
    }
    lines[lines.length - 1] =
        ' ' + lines[lines.length - 1] + ']' + (isLast ? '' : newLineSep);
    return lines;
}
function createComplexTuples(vals) {
    const complexTuples = [];
    for (let i = 0; i < vals.length; i += 2) {
        complexTuples.push([vals[i], vals[i + 1]]);
    }
    return complexTuples;
}

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
/**
 * A mutable object, similar to `tf.Tensor`, that allows users to set values
 * at locations before converting to an immutable `tf.Tensor`.
 *
 * See `tf.buffer` for creating a tensor buffer.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
class TensorBuffer {
    constructor(shape, dtype, values) {
        this.dtype = dtype;
        this.shape = shape.slice();
        this.size = sizeFromShape(shape);
        if (values != null) {
            const n = values.length;
            assert(n === this.size, () => `Length of values '${n}' does not match the size ` +
                `inferred by the shape '${this.size}'.`);
        }
        if (dtype === 'complex64') {
            throw new Error(`complex64 dtype TensorBuffers are not supported. Please create ` +
                `a TensorBuffer for the real and imaginary parts separately and ` +
                `call tf.complex(real, imag).`);
        }
        this.values = values || getArrayFromDType(dtype, this.size);
        this.strides = computeStrides(shape);
    }
    /**
     * Sets a value in the buffer at a given location.
     *
     * @param value The value to set.
     * @param locs  The location indices.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    set(value, ...locs) {
        if (locs.length === 0) {
            locs = [0];
        }
        assert(locs.length === this.rank, () => `The number of provided coordinates (${locs.length}) must ` +
            `match the rank (${this.rank})`);
        const index = this.locToIndex(locs);
        this.values[index] = value;
    }
    /**
     * Returns the value in the buffer at the provided location.
     *
     * @param locs The location indices.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    get(...locs) {
        if (locs.length === 0) {
            locs = [0];
        }
        let i = 0;
        for (const loc of locs) {
            if (loc < 0 || loc >= this.shape[i]) {
                const msg = `Requested out of range element at ${locs}. ` +
                    `  Buffer shape=${this.shape}`;
                throw new Error(msg);
            }
            i++;
        }
        let index = locs[locs.length - 1];
        for (let i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.values[index];
    }
    locToIndex(locs) {
        if (this.rank === 0) {
            return 0;
        }
        else if (this.rank === 1) {
            return locs[0];
        }
        let index = locs[locs.length - 1];
        for (let i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    }
    indexToLoc(index) {
        if (this.rank === 0) {
            return [];
        }
        else if (this.rank === 1) {
            return [index];
        }
        const locs = new Array(this.shape.length);
        for (let i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    }
    get rank() {
        return this.shape.length;
    }
    /**
     * Creates an immutable `tf.Tensor` object from the buffer.
     *
     * @doc {heading: 'Tensors', subheading: 'Creation'}
     */
    toTensor() {
        return trackerFn().makeTensor(this.values, this.shape, this.dtype);
    }
}
// For tracking tensor creation and disposal.
let trackerFn = null;
// Used by chaining methods to call into ops.
let opHandler = null;
/**
 * An external consumer can register itself as the tensor tracker. This way
 * the Tensor class can notify the tracker for every tensor created and
 * disposed.
 */
function setTensorTracker(fn) {
    trackerFn = fn;
}
/**
 * An external consumer can register itself as the op handler. This way the
 * Tensor class can have chaining methods that call into ops via the op
 * handler.
 */
function setOpHandler(handler) {
    opHandler = handler;
}
/**
 * A `tf.Tensor` object represents an immutable, multidimensional array of
 * numbers that has a shape and a data type.
 *
 * See `tf.tensor` for details on how to create a `tf.Tensor`.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
class Tensor {
    constructor(shape, dtype, dataId, id) {
        /** Whether this tensor has been globally kept. */
        this.kept = false;
        this.isDisposedInternal = false;
        this.shape = shape.slice();
        this.dtype = dtype || 'float32';
        this.size = sizeFromShape(shape);
        this.strides = computeStrides(shape);
        this.dataId = dataId;
        this.id = id;
        this.rankType = (this.rank < 5 ? this.rank.toString() : 'higher');
    }
    get rank() {
        return this.shape.length;
    }
    /**
     * Returns a promise of `tf.TensorBuffer` that holds the underlying data.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async buffer() {
        const vals = await this.data();
        return opHandler.buffer(this.shape, this.dtype, vals);
    }
    /**
     * Returns a `tf.TensorBuffer` that holds the underlying data.
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    bufferSync() {
        return opHandler.buffer(this.shape, this.dtype, this.dataSync());
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * asynchronously.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async array() {
        const vals = await this.data();
        return toNestedArray(this.shape, vals);
    }
    /**
     * Returns the tensor data as a nested array. The transfer of data is done
     * synchronously.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    arraySync() {
        return toNestedArray(this.shape, this.dataSync());
    }
    /**
     * Asynchronously downloads the values from the `tf.Tensor`. Returns a
     * promise of `TypedArray` that resolves when the computation has finished.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    async data() {
        this.throwIfDisposed();
        const data = trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            const bytes = await data;
            try {
                return bytes.map(b => decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /**
     * Synchronously downloads the values from the `tf.Tensor`. This blocks the
     * UI thread until the values are ready, which can cause performance issues.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    dataSync() {
        this.throwIfDisposed();
        const data = trackerFn().readSync(this.dataId);
        if (this.dtype === 'string') {
            try {
                return data.map(b => decodeString(b));
            }
            catch (_a) {
                throw new Error('Failed to decode the string bytes into utf-8. ' +
                    'To get the original bytes, call tensor.bytes().');
            }
        }
        return data;
    }
    /** Returns the underlying bytes of the tensor's data. */
    async bytes() {
        this.throwIfDisposed();
        const data = await trackerFn().read(this.dataId);
        if (this.dtype === 'string') {
            return data;
        }
        else {
            return new Uint8Array(data.buffer);
        }
    }
    /**
     * Disposes `tf.Tensor` from memory.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    dispose() {
        if (this.isDisposed) {
            return;
        }
        trackerFn().disposeTensor(this);
        this.isDisposedInternal = true;
    }
    get isDisposed() {
        return this.isDisposedInternal;
    }
    throwIfDisposed() {
        if (this.isDisposed) {
            throw new Error(`Tensor is disposed.`);
        }
    }
    /**
     * Prints the `tf.Tensor`. See `tf.print` for details.
     *
     * @param verbose Whether to print verbose information about the tensor,
     *    including dtype and size.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    print(verbose = false) {
        return opHandler.print(this, verbose);
    }
    /**
     * Returns a copy of the tensor. See `tf.clone` for details.
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    clone() {
        this.throwIfDisposed();
        return opHandler.clone(this);
    }
    /**
     * Returns a human-readable description of the tensor. Useful for logging.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    toString(verbose = false) {
        const vals = this.dataSync();
        return tensorToString(vals, this.shape, this.dtype, verbose);
    }
    cast(dtype) {
        this.throwIfDisposed();
        return opHandler.cast(this, dtype);
    }
    variable(trainable = true, name, dtype) {
        this.throwIfDisposed();
        return trackerFn().makeVariable(this, trainable, name, dtype);
    }
}
Object.defineProperty(Tensor, Symbol.hasInstance, {
    value: (instance) => {
        // Implementation note: we should use properties of the object that will be
        // defined before the constructor body has finished executing (methods).
        // This is because when this code is transpiled by babel, babel will call
        // classCallCheck before the constructor body is run.
        // See https://github.com/tensorflow/tfjs/issues/3384 for backstory.
        return !!instance && instance.data != null && instance.dataSync != null &&
            instance.throwIfDisposed != null;
    }
});
function getGlobalTensorClass() {
    // Use getGlobal so that we can augment the Tensor class across package
    // boundaries becase the node resolution alg may result in different modules
    // being returned for this file depending on the path they are loaded from.
    return getGlobal('Tensor', () => {
        return Tensor;
    });
}
// Global side effect. Cache global reference to Tensor class
getGlobalTensorClass();
/**
 * A mutable `tf.Tensor`, useful for persisting state, e.g. for training.
 *
 * @doc {heading: 'Tensors', subheading: 'Classes'}
 */
class Variable extends Tensor {
    constructor(initialValue, trainable, name, tensorId) {
        super(initialValue.shape, initialValue.dtype, initialValue.dataId, tensorId);
        this.trainable = trainable;
        this.name = name;
    }
    /**
     * Assign a new `tf.Tensor` to this variable. The new `tf.Tensor` must have
     * the same shape and dtype as the old `tf.Tensor`.
     *
     * @param newValue New tensor to be assigned to this variable.
     *
     * @doc {heading: 'Tensors', subheading: 'Classes'}
     */
    assign(newValue) {
        if (newValue.dtype !== this.dtype) {
            throw new Error(`dtype of the new value (${newValue.dtype}) and ` +
                `previous value (${this.dtype}) must match`);
        }
        if (!arraysEqual(newValue.shape, this.shape)) {
            throw new Error(`shape of the new value (${newValue.shape}) and ` +
                `previous value (${this.shape}) must match`);
        }
        trackerFn().disposeTensor(this);
        this.dataId = newValue.dataId;
        trackerFn().incRef(this, null /* backend */);
    }
    dispose() {
        trackerFn().disposeVariable(this);
        this.isDisposedInternal = true;
    }
}
Object.defineProperty(Variable, Symbol.hasInstance, {
    value: (instance) => {
        return instance instanceof Tensor && instance.assign != null &&
            instance.assign instanceof Function;
    }
});

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
var Rank;
(function (Rank) {
    Rank["R0"] = "R0";
    Rank["R1"] = "R1";
    Rank["R2"] = "R2";
    Rank["R3"] = "R3";
    Rank["R4"] = "R4";
    Rank["R5"] = "R5";
    Rank["R6"] = "R6";
})(Rank || (Rank = {}));
// Looks for upcasting types. Used, for example, in operations with mixed dtype
// inputs.
var UpcastInt32AndMap;
(function (UpcastInt32AndMap) {
    UpcastInt32AndMap["float32"] = "float32";
    UpcastInt32AndMap["int32"] = "int32";
    UpcastInt32AndMap["bool"] = "int32";
    UpcastInt32AndMap["complex64"] = "complex64";
})(UpcastInt32AndMap || (UpcastInt32AndMap = {}));
var UpcastBoolAndMap;
(function (UpcastBoolAndMap) {
    UpcastBoolAndMap["float32"] = "float32";
    UpcastBoolAndMap["int32"] = "int32";
    UpcastBoolAndMap["bool"] = "bool";
    UpcastBoolAndMap["complex64"] = "complex64";
})(UpcastBoolAndMap || (UpcastBoolAndMap = {}));
var UpcastFloat32AndMap;
(function (UpcastFloat32AndMap) {
    UpcastFloat32AndMap["float32"] = "float32";
    UpcastFloat32AndMap["int32"] = "float32";
    UpcastFloat32AndMap["bool"] = "float32";
    UpcastFloat32AndMap["complex64"] = "complex64";
})(UpcastFloat32AndMap || (UpcastFloat32AndMap = {}));
var UpcastComplex64AndMap;
(function (UpcastComplex64AndMap) {
    UpcastComplex64AndMap["float32"] = "complex64";
    UpcastComplex64AndMap["int32"] = "complex64";
    UpcastComplex64AndMap["bool"] = "complex64";
    UpcastComplex64AndMap["complex64"] = "complex64";
})(UpcastComplex64AndMap || (UpcastComplex64AndMap = {}));
const upcastTypeMap = {
    'float32': UpcastFloat32AndMap,
    'int32': UpcastInt32AndMap,
    'bool': UpcastBoolAndMap,
    'complex64': UpcastComplex64AndMap
};
function upcastType(typeA, typeB) {
    if (typeA === 'string' || typeB === 'string') {
        if (typeA === 'string' && typeB === 'string') {
            return 'string';
        }
        throw new Error(`Can not upcast ${typeA} with ${typeB}`);
    }
    return upcastTypeMap[typeA][typeB];
}
/** Returns the output type after summation. */
function sumOutType(type) {
    return upcastType(type, 'int32');
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
function makeTypesMatch(a, b) {
    if (a.dtype === b.dtype) {
        return [a, b];
    }
    const dtype = upcastType(a.dtype, b.dtype);
    return [a.cast(dtype), b.cast(dtype)];
}
function assertTypesMatch(a, b) {
    assert(a.dtype === b.dtype, () => `The dtypes of the first(${a.dtype}) and` +
        ` second(${b.dtype}) input must match`);
}
function isTensorInList(tensor, tensorList) {
    return tensorList.some(x => x.id === tensor.id);
}
/**
 * Extracts any `Tensor`s found within the provided object.
 *
 * @param container an object that may be a `Tensor` or may directly contain
 *   `Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. In general it
 *   is safe to pass any object here, except that `Promise`s are not
 *   supported.
 * @returns An array of `Tensors` found within the passed object. If the
 *   argument is simply a `Tensor', a list containing that `Tensor` is
 *   returned. If the object is not a `Tensor` or does not
 *   contain `Tensors`, an empty list is returned.
 */
function getTensorsInContainer(result) {
    const list = [];
    const seen = new Set();
    walkTensorContainer(result, list, seen);
    return list;
}
function walkTensorContainer(container, list, seen) {
    if (container == null) {
        return;
    }
    if (container instanceof Tensor) {
        list.push(container);
        return;
    }
    if (!isIterable(container)) {
        return;
    }
    // Iteration over keys works also for arrays.
    const iterable = container;
    for (const k in iterable) {
        const val = iterable[k];
        if (!seen.has(val)) {
            seen.add(val);
            walkTensorContainer(val, list, seen);
        }
    }
}
// tslint:disable-next-line:no-any
function isIterable(obj) {
    return Array.isArray(obj) || typeof obj === 'object';
}

var tensor_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    makeTypesMatch: makeTypesMatch,
    assertTypesMatch: assertTypesMatch,
    isTensorInList: isTensorInList,
    getTensorsInContainer: getTensorsInContainer
});

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
function isRegisteredKernelInvocation(kernelInvocation) {
    return kernelInvocation.kernelName != null;
}
class EngineState {
    constructor() {
        // Public since optimizers will use it.
        this.registeredVariables = {};
        this.nextTapeNodeId = 0;
        this.numBytes = 0;
        this.numTensors = 0;
        this.numStringTensors = 0;
        this.numDataBuffers = 0;
        // Number of nested tf.grad() statements when computing higher-order
        // gradients. E.g. `1` for first-order gradients and `2` for second-order
        // gradients. Used to track if the tape should be removed after a backprop.
        this.gradientDepth = 0;
        // Number of nested kernel calls. When kernel depth is greater than 1, we turn
        // off the tape.
        this.kernelDepth = 0;
        this.scopeStack = [];
        /**
         * Keeps track of the number of data moves during a kernel execution. We
         * maintain a stack since kernels can call other kernels, recursively.
         */
        this.numDataMovesStack = [];
        this.nextScopeId = 0;
        this.tensorInfo = new WeakMap();
        this.profiling = false;
        this.activeProfile = {
            newBytes: 0,
            newTensors: 0,
            peakBytes: 0,
            kernels: [],
            result: null,
            get kernelNames() {
                return Array.from(new Set(this.kernels.map(k => k.name)));
            }
        };
    }
    dispose() {
        for (const variableName in this.registeredVariables) {
            this.registeredVariables[variableName].dispose();
        }
    }
}
class Engine {
    constructor(ENV) {
        this.ENV = ENV;
        this.registry = {};
        this.registryFactory = {};
        this.pendingBackendInitId = 0;
        this.state = new EngineState();
    }
    async ready() {
        if (this.pendingBackendInit != null) {
            return this.pendingBackendInit.then(() => { });
        }
        if (this.backendInstance != null) {
            return;
        }
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const success = await this.initializeBackend(backendName).success;
            if (success) {
                await this.setBackend(backendName);
                return;
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations ` +
            `failed.`);
    }
    get backend() {
        if (this.pendingBackendInit != null) {
            throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make ` +
                `sure to await tf.ready() or await tf.setBackend() before calling ` +
                `other methods`);
        }
        if (this.backendInstance == null) {
            const { name, asyncInit } = this.initializeBackendsAndReturnBest();
            if (asyncInit) {
                throw new Error(`The highest priority backend '${name}' has not yet been ` +
                    `initialized. Make sure to await tf.ready() or ` +
                    `await tf.setBackend() before calling other methods`);
            }
            this.setBackend(name);
        }
        return this.backendInstance;
    }
    backendNames() {
        return Object.keys(this.registryFactory);
    }
    findBackend(backendName) {
        if (!(backendName in this.registry)) {
            // If the backend hasn't been initialized but we have a registry entry for
            // it, initialize it and return it.
            if (backendName in this.registryFactory) {
                const { asyncInit } = this.initializeBackend(backendName);
                if (asyncInit) {
                    // Backend is not ready yet.
                    return null;
                }
            }
            else {
                return null;
            }
        }
        return this.registry[backendName];
    }
    findBackendFactory(backendName) {
        if (!(backendName in this.registryFactory)) {
            return null;
        }
        return this.registryFactory[backendName].factory;
    }
    registerBackend(backendName, factory, priority = 1) {
        if (backendName in this.registryFactory) {
            console.warn(`${backendName} backend was already registered. ` +
                `Reusing existing backend factory.`);
            return false;
        }
        this.registryFactory[backendName] = { factory, priority };
        return true;
    }
    async setBackend(backendName) {
        if (this.registryFactory[backendName] == null) {
            throw new Error(`Backend name '${backendName}' not found in registry`);
        }
        this.backendName = backendName;
        if (this.registry[backendName] == null) {
            this.backendInstance = null;
            const { success, asyncInit } = this.initializeBackend(backendName);
            const result = asyncInit ? await success : success;
            if (!result) {
                return false;
            }
        }
        this.backendInstance = this.registry[backendName];
        this.setupRegisteredKernels();
        // Reset the profiler.
        this.profiler = new Profiler(this.backendInstance);
        return true;
    }
    setupRegisteredKernels() {
        const kernels = getKernelsForBackend(this.backendName);
        kernels.forEach(kernel => {
            if (kernel.setupFunc != null) {
                kernel.setupFunc(this.backendInstance);
            }
        });
    }
    disposeRegisteredKernels(backendName) {
        const kernels = getKernelsForBackend(backendName);
        kernels.forEach(kernel => {
            if (kernel.disposeFunc != null) {
                kernel.disposeFunc(this.registry[backendName]);
            }
        });
    }
    /**
     * Initializes a backend by looking up the backend name in the factory
     * registry and calling the factory method. Returns a boolean representing
     * whether the initialization of the backend suceeded. Throws an error if
     * there is no backend in the factory registry.
     */
    initializeBackend(backendName) {
        const registryFactoryEntry = this.registryFactory[backendName];
        if (registryFactoryEntry == null) {
            throw new Error(`Cannot initialize backend ${backendName}, no registration found.`);
        }
        try {
            const backend = registryFactoryEntry.factory();
            /* Test if the factory returns a promise.
            Done in a more liberal way than
            previous 'Promise.resolve(backend)===backend'
            as we needed to account for custom Promise
            implementations (e.g. Angular) */
            if (backend && !(backend instanceof KernelBackend) &&
                typeof backend.then === 'function') {
                const promiseId = ++this.pendingBackendInitId;
                const success = backend
                    .then(backendInstance => {
                    // Outdated promise. Another backend was set in the meantime.
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.registry[backendName] = backendInstance;
                    this.pendingBackendInit = null;
                    return true;
                })
                    .catch(err => {
                    // Outdated promise. Another backend was set in the meantime.
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.pendingBackendInit = null;
                    console.warn(`Initialization of backend ${backendName} failed`);
                    console.warn(err.stack || err.message);
                    return false;
                });
                this.pendingBackendInit = success;
                return { success, asyncInit: true };
            }
            else {
                this.registry[backendName] = backend;
                return { success: true, asyncInit: false };
            }
        }
        catch (err) {
            console.warn(`Initialization of backend ${backendName} failed`);
            console.warn(err.stack || err.message);
            return { success: false, asyncInit: false };
        }
    }
    removeBackend(backendName) {
        if (!(backendName in this.registryFactory)) {
            throw new Error(`${backendName} backend not found in registry`);
        }
        if (this.backendName === backendName && this.pendingBackendInit != null) {
            // There is a pending promise of the backend we want to remove. Make it
            // obsolete.
            this.pendingBackendInitId++;
        }
        if (backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        delete this.registryFactory[backendName];
        // Unset the backend if it is active.
        if (this.backendName === backendName) {
            this.pendingBackendInit = null;
            this.backendName = null;
            this.backendInstance = null;
        }
    }
    getSortedBackends() {
        if (Object.keys(this.registryFactory).length === 0) {
            throw new Error('No backend found in registry.');
        }
        return Object.keys(this.registryFactory).sort((a, b) => {
            // Highest priority comes first.
            return this.registryFactory[b].priority -
                this.registryFactory[a].priority;
        });
    }
    initializeBackendsAndReturnBest() {
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const { success, asyncInit } = this.initializeBackend(backendName);
            if (asyncInit || success) {
                return { name: backendName, asyncInit };
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations ` +
            `failed.`);
    }
    moveData(backend, dataId) {
        const info = this.state.tensorInfo.get(dataId);
        const srcBackend = info.backend;
        const values = this.readSync(dataId);
        // Delete the tensor from the old backend and move it to the new
        // backend.
        srcBackend.disposeData(dataId);
        info.backend = backend;
        backend.move(dataId, values, info.shape, info.dtype);
        if (this.shouldCheckForMemLeaks()) {
            // Track the number of moves during a kernel execution to correctly
            // detect memory leaks.
            this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1]++;
        }
    }
    tidy(nameOrFn, fn) {
        let name = null;
        if (fn == null) {
            // Called with only 1 argument.
            if (typeof nameOrFn !== 'function') {
                throw new Error('Please provide a function to tidy()');
            }
            fn = nameOrFn;
        }
        else {
            // Called with 2 arguments.
            if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
                throw new Error('When calling with two arguments, the first argument ' +
                    'to tidy() must be a string');
            }
            if (typeof fn !== 'function') {
                throw new Error('When calling with two arguments, the 2nd argument ' +
                    'to tidy() must be a function');
            }
            name = nameOrFn;
            // TODO(nsthorat,smilkov): Do operation logging and performance
            // profiling.
        }
        let result;
        return this.scopedRun(() => this.startScope(name), () => this.endScope(result), () => {
            result = fn();
            if (result instanceof Promise) {
                console.error('Cannot return a Promise inside of tidy.');
            }
            return result;
        });
    }
    scopedRun(start, end, f) {
        start();
        try {
            const res = f();
            end();
            return res;
        }
        catch (ex) {
            end();
            throw ex;
        }
    }
    nextTensorId() {
        return Engine.nextTensorId++;
    }
    nextVariableId() {
        return Engine.nextVariableId++;
    }
    /**
     * This method is called instead of the public-facing tensor.clone() when
     * saving a tensor for backwards pass. It makes sure to add the clone
     * operation to the tape regardless of being called inside a kernel
     * execution.
     */
    clone(x) {
        const y = this.makeTensorFromDataId(x.dataId, x.shape, x.dtype);
        const inputs = { x };
        const grad = (dy) => ({
            x: () => {
                const dtype = 'float32';
                const gradInputs = { x: dy };
                const attrs = { dtype };
                return ENGINE.runKernel(Cast, gradInputs, 
                // tslint:disable-next-line: no-unnecessary-type-assertion
                attrs);
            }
        });
        const saved = [];
        this.addTapeNode(this.state.activeScope.name, inputs, [y], grad, saved, {});
        return y;
    }
    /**
     * Execute a kernel with the given name and return the output tensor.
     *
     * @param kernelName The name of the kernel to execute.
     * @param inputs A map of input names to tensors.
     * @param attrs A map of attribute names to their values. An attribute is a
     *     primitive (non-tensor) input to the kernel.
     * @param inputsToSave A list of tensors, inputs to save for the backprop
     *     computation.
     * @param outputsToSave A list of booleans, specifying which output to save
     *     for the backprop computation. These are booleans since the output
     * tensors are not visible to the user.
     */
    runKernel(kernelName, inputs, attrs) {
        const hasKernel = getKernel(kernelName, this.backendName) != null;
        if (!hasKernel) {
            throw new Error(`Kernel '${kernelName}' not registered for backend '${this.backendName}'`);
        }
        return this.runKernelFunc({ kernelName, inputs, attrs });
    }
    shouldCheckForMemLeaks() {
        return this.ENV.getBool('IS_TEST');
    }
    checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos) {
        const numDataIdsAfter = this.backend.numDataIds();
        // Count the number of data ids associated with the result of the kernel.
        let numOutputDataIds = 0;
        outInfos.forEach(info => {
            // Complex numbers allocate 3 data ids, one for 'real', one for
            // 'imaginary', and one for the container that holds the former two.
            numOutputDataIds += (info.dtype === 'complex64' ? 3 : 1);
        });
        // Account for the number of moves during kernel execution. A "data move"
        // can happen in the middle of a kernel execution, placing a new (key,value)
        // pair in the data storage. Since data moves have net zero effect (we
        // always remove the data from the old backend), we have to cancel them out
        // when detecting memory leaks.
        const numMoves = this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1];
        const dataIdsLeaked = numDataIdsAfter - numDataIdsBefore - numOutputDataIds - numMoves;
        if (dataIdsLeaked > 0) {
            throw new Error(`Backend '${this.backendName}' has an internal memory leak ` +
                `(${dataIdsLeaked} data ids) after running '${kernelName}'`);
        }
    }
    /**
     * Internal helper method to execute a kernel Func
     *
     * Use `runKernel` to execute kernels from outside of engine.
     */
    runKernelFunc(kernelParams) {
        let outputs;
        let saved = [];
        const isTapeOn = this.isTapeOn();
        const startingBytecount = this.state.numBytes;
        const startingNumTensors = this.state.numTensors;
        if (this.shouldCheckForMemLeaks()) {
            this.state.numDataMovesStack.push(0);
        }
        let kernelFunc;
        if (this.backendName == null) {
            // backend has not been initialized yet (backend initialization is lazy
            // can be deferred until an op/ kernel is run).
            // The below getter has side effects that will try to initialize the
            // backend and set properties like this.backendName
            // tslint:disable-next-line: no-unused-expression
            this.backend;
        }
        let out;
        const kernelOrScopeName = isRegisteredKernelInvocation(kernelParams) ?
            kernelParams.kernelName :
            this.state.activeScope != null ? this.state.activeScope.name : '';
        // Create the kernelFunc from either a registered kernel OR passed in
        // forward/backward functions (used by custom grad). In this context a
        // kernelFunc wraps a kernel implementation with some bookkeeping.
        if (isRegisteredKernelInvocation(kernelParams)) {
            const { kernelName, inputs, attrs } = kernelParams;
            if (this.backendName == null) {
                // backend has not been initialized yet (backend initialization is lazy
                // can be deferred until an op/ kernel is run).
                // The below getter has side effects that will try to initialize the
                // backend and set properties like this.backendName
                // tslint:disable-next-line: no-unused-expression
                this.backend;
            }
            const kernel = getKernel(kernelName, this.backendName);
            assert(kernel != null, () => `Cannot find registered kernel '${kernelName}' for backend '${this.backendName}'`);
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = kernel.kernelFunc({ inputs, attrs, backend: this.backend });
                const outInfos = Array.isArray(out) ? out : [out];
                if (this.shouldCheckForMemLeaks()) {
                    this.checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos);
                }
                const outTensors = outInfos.map((outInfo) => {
                    // todo (yassogba) remove this option (Tensor) when node backend
                    // methods have been modularized and they all return tensorInfo.
                    // TensorInfos do not have a rank attribute.
                    if (outInfo.rank != null) {
                        return outInfo;
                    }
                    const { dataId, shape, dtype } = outInfo;
                    return this.makeTensorFromDataId(dataId, shape, dtype);
                });
                // Save any required inputs and outputs.
                // Do not save unless we are recording to the tape. Otherwise it would
                // cause a mem leak since there would be no backprop for these tensors
                // (which would otherwise dispose them).
                if (isTapeOn) {
                    const tensorsToSave = this.getTensorsForGradient(kernelName, inputs, outTensors);
                    saved = this.saveTensorsForBackwardMode(tensorsToSave);
                }
                return outTensors;
            };
        }
        else {
            const { forwardFunc } = kernelParams;
            // Running a customGrad op.
            const saveFunc = (tensors) => {
                // Do not save unless we are recording to the tape. Otherwise it would
                // cause a mem leak since we would never run backprop, which disposes
                // the kept tensors.
                if (!isTapeOn) {
                    return;
                }
                saved = tensors.map(tensor => this.keep(this.clone(tensor)));
            };
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = this.tidy(() => forwardFunc(this.backend, saveFunc));
                const outs = (Array.isArray(out) ? out : [out]);
                if (this.shouldCheckForMemLeaks()) {
                    // Scope name is used to print a more helpful error message if needed.
                    this.checkKernelForMemLeak(kernelOrScopeName, numDataIdsBefore, outs);
                }
                return outs;
            };
        }
        //
        // Run the kernelFunc. Optionally profiling it.
        //
        const { inputs, attrs } = kernelParams;
        const backwardsFunc = isRegisteredKernelInvocation(kernelParams) ?
            null :
            kernelParams.backwardsFunc;
        let kernelProfile;
        this.scopedRun(
        // Stop recording to a tape when running a kernel.
        () => this.state.kernelDepth++, () => this.state.kernelDepth--, () => {
            if (!this.ENV.getBool('DEBUG') && !this.state.profiling) {
                outputs = kernelFunc();
            }
            else {
                kernelProfile = this.profiler.profileKernel(kernelOrScopeName, inputs, () => kernelFunc());
                if (this.ENV.getBool('DEBUG')) {
                    this.profiler.logKernelProfile(kernelProfile);
                }
                outputs = kernelProfile.outputs;
            }
        });
        if (isTapeOn) {
            this.addTapeNode(kernelOrScopeName, inputs, outputs, backwardsFunc, saved, attrs);
        }
        if (this.state.profiling) {
            this.state.activeProfile.kernels.push({
                name: kernelOrScopeName,
                bytesAdded: this.state.numBytes - startingBytecount,
                totalBytesSnapshot: this.state.numBytes,
                tensorsAdded: this.state.numTensors - startingNumTensors,
                totalTensorsSnapshot: this.state.numTensors,
                inputShapes: Object.keys(inputs).map(key => inputs[key] != null ? inputs[key].shape : null),
                outputShapes: outputs.map(item => item.shape),
                kernelTimeMs: kernelProfile.timeMs,
                extraInfo: kernelProfile.extraInfo
            });
        }
        return (Array.isArray(out) ? outputs : outputs[0]);
    }
    /**
     * Saves tensors used in forward mode for use in backward mode.
     *
     * @param tensors the list of tensors to save.
     */
    saveTensorsForBackwardMode(tensors) {
        const saved = tensors.map(tensor => this.keep(this.clone(tensor)));
        return saved;
    }
    /**
     * Returns a list of tensors to save for a given gradient calculation.
     *
     * @param kernelName name of kernel to look up gradient for.
     * @param inputs a map of input tensors.
     * @param outputs an array of output tensors from forward mode of kernel.
     */
    getTensorsForGradient(kernelName, inputs, outputs) {
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            const inputsToSave = gradConfig.inputsToSave || [];
            const outputsToSave = gradConfig.outputsToSave || [];
            // If saveAllInputs is true, all inputs will be saved. Otherwise, inputs
            // specified in inputsToSave will be saved.
            let inputTensorsToSave;
            if (gradConfig.saveAllInputs) {
                assert(Array.isArray(inputs), () => 'saveAllInputs is true, expected inputs to be an array.');
                inputTensorsToSave = Object.keys(inputs).map((key) => inputs[key]);
            }
            else {
                inputTensorsToSave = inputsToSave.map((inputName) => inputs[inputName]);
            }
            const outputTensorsToSave = outputs.filter((_, i) => outputsToSave[i]);
            return inputTensorsToSave.concat(outputTensorsToSave);
        }
        // We return an empty list rather than throw an error because the kernel we
        // are looking up may not actually be relevant to backproping through the
        // overall function
        //
        // See 'does not error if irrelevant (pruned) ops are missing grads' test
        // in gradients_test.ts for an example.
        return [];
    }
    /**
     * Internal method used by public APIs for tensor creation. Makes a new
     * tensor with the provided shape, dtype and values. It always
     * creates a new data id and writes the values to the underlying backend.
     */
    makeTensor(values, shape, dtype, backend) {
        if (values == null) {
            throw new Error('Values passed to engine.makeTensor() are null');
        }
        dtype = dtype || 'float32';
        backend = backend || this.backend;
        let backendVals = values;
        if (dtype === 'string' && isString(values[0])) {
            backendVals = values.map(d => encodeString(d));
        }
        const dataId = backend.write(backendVals, shape, dtype);
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.incRef(t, backend);
        // Count bytes for string tensors.
        if (dtype === 'string') {
            const info = this.state.tensorInfo.get(dataId);
            const newBytes = bytesFromStringArray(backendVals);
            this.state.numBytes += newBytes - info.bytes;
            info.bytes = newBytes;
        }
        return t;
    }
    /**
     * Internal method used by backends. Makes a new tensor
     * that is a wrapper around an existing data id. It doesn't create
     * a new data id, only increments the ref count used in memory tracking.
     */
    makeTensorFromDataId(dataId, shape, dtype, backend) {
        dtype = dtype || 'float32';
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.incRef(t, backend);
        return t;
    }
    makeVariable(initialValue, trainable = true, name, dtype) {
        name = name || this.nextVariableId().toString();
        if (dtype != null && dtype !== initialValue.dtype) {
            initialValue = initialValue.cast(dtype);
        }
        const v = new Variable(initialValue, trainable, name, this.nextTensorId());
        if (this.state.registeredVariables[v.name] != null) {
            throw new Error(`Variable with name ${v.name} was already registered`);
        }
        this.state.registeredVariables[v.name] = v;
        this.incRef(v, this.backend);
        return v;
    }
    incRef(a, backend) {
        const refCount = this.state.tensorInfo.has(a.dataId) ?
            this.state.tensorInfo.get(a.dataId).refCount :
            0;
        this.state.numTensors++;
        if (a.dtype === 'string') {
            this.state.numStringTensors++;
        }
        if (refCount === 0) {
            this.state.numDataBuffers++;
            // Bytes for complex numbers are counted by their components. Bytes for
            // string tensors are counted when writing values.
            let bytes = 0;
            if (a.dtype !== 'complex64' && a.dtype !== 'string') {
                bytes = a.size * bytesPerElement(a.dtype);
            }
            this.state.tensorInfo.set(a.dataId, {
                backend: backend || this.backend,
                dtype: a.dtype,
                shape: a.shape,
                bytes,
                refCount: 0
            });
            this.state.numBytes += bytes;
        }
        this.state.tensorInfo.get(a.dataId).refCount++;
        if (!(a instanceof Variable)) {
            this.track(a);
        }
    }
    disposeTensor(a) {
        if (!this.state.tensorInfo.has(a.dataId)) {
            return;
        }
        this.state.numTensors--;
        if (a.dtype === 'string') {
            this.state.numStringTensors--;
        }
        const info = this.state.tensorInfo.get(a.dataId);
        const refCount = info.refCount;
        if (refCount <= 1) {
            // Don't count bytes for complex numbers as they are counted by their
            // components.
            if (a.dtype !== 'complex64') {
                this.state.numBytes -= info.bytes;
            }
            this.state.numDataBuffers--;
            info.backend.disposeData(a.dataId);
            this.state.tensorInfo.delete(a.dataId);
        }
        else {
            // Notify the backend to descrease the ref count for complex tensor
            // components. This method is only implemented in WebGL right now. When
            // there are multiple references, complex tensor cannot dispose the
            // components if ref count is not in sync with engine.
            info.backend.decComplexRef(a.dataId);
            this.state.tensorInfo.get(a.dataId).refCount--;
        }
        // TODO(nsthorat): Construct an error and save the stack trace for
        // debugging when in debug mode. Creating a stack trace is too expensive
        // to do unconditionally.
    }
    disposeVariables() {
        for (const varName in this.state.registeredVariables) {
            const v = this.state.registeredVariables[varName];
            this.disposeVariable(v);
        }
    }
    disposeVariable(v) {
        this.disposeTensor(v);
        if (this.state.registeredVariables[v.name] != null) {
            delete this.state.registeredVariables[v.name];
        }
    }
    memory() {
        const info = this.backend.memory();
        info.numTensors = this.state.numTensors;
        info.numDataBuffers = this.state.numDataBuffers;
        info.numBytes = this.state.numBytes;
        if (this.state.numStringTensors > 0) {
            info.unreliable = true;
            if (info.reasons == null) {
                info.reasons = [];
            }
            info.reasons.push('Memory usage by string tensors is approximate ' +
                '(2 bytes per character)');
        }
        return info;
    }
    async profile(query) {
        this.state.profiling = true;
        const startBytes = this.state.numBytes;
        const startNumTensors = this.state.numTensors;
        this.state.activeProfile.kernels = [];
        this.state.activeProfile.result = await query();
        this.state.profiling = false;
        this.state.activeProfile.peakBytes = Math.max(...this.state.activeProfile.kernels.map(d => d.totalBytesSnapshot));
        this.state.activeProfile.newBytes = this.state.numBytes - startBytes;
        this.state.activeProfile.newTensors =
            this.state.numTensors - startNumTensors;
        for (const kernel of this.state.activeProfile.kernels) {
            kernel.kernelTimeMs = await kernel.kernelTimeMs;
            kernel.extraInfo = await kernel.extraInfo;
        }
        return this.state.activeProfile;
    }
    isTapeOn() {
        return this.state.gradientDepth > 0 && this.state.kernelDepth === 0;
    }
    addTapeNode(kernelName, inputs, outputs, gradientsFunc, saved, attrs) {
        const tapeNode = { id: this.state.nextTapeNodeId++, kernelName, inputs, outputs, saved };
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            gradientsFunc = gradConfig.gradFunc;
        }
        if (gradientsFunc != null) {
            tapeNode.gradient = (dys) => {
                // TODO(smilkov): To optimize back-prop, pass dys that are not used in
                // the backprop graph to the user as null instead of zeros
                dys = dys.map((dy, i) => {
                    if (dy == null) {
                        const output = outputs[i];
                        const vals = makeZerosTypedArray(output.size, output.dtype);
                        return this.makeTensor(vals, output.shape, output.dtype);
                    }
                    return dy;
                });
                // Grad functions of ops with single outputs expect a dy, while ops
                // with multiple outputs expect dys (array of dy).
                return gradientsFunc(dys.length > 1 ? dys : dys[0], saved, attrs);
            };
        }
        this.state.activeTape.push(tapeNode);
    }
    keep(result) {
        result.kept = true;
        return result;
    }
    startTape() {
        if (this.state.gradientDepth === 0) {
            this.state.activeTape = [];
        }
        this.state.gradientDepth++;
    }
    endTape() {
        this.state.gradientDepth--;
    }
    /**
     * Start a scope. Use this with endScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    startScope(name) {
        const scopeInfo = {
            track: [],
            name: 'unnamed scope',
            id: this.state.nextScopeId++
        };
        if (name) {
            scopeInfo.name = name;
        }
        this.state.scopeStack.push(scopeInfo);
        this.state.activeScope = scopeInfo;
    }
    /**
     * End a scope. Use this with startScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    endScope(result) {
        const tensorsToTrackInParent = getTensorsInContainer(result);
        const tensorsToTrackInParentSet = new Set(tensorsToTrackInParent.map(t => t.id));
        // Dispose the arrays tracked in this scope.
        for (let i = 0; i < this.state.activeScope.track.length; i++) {
            const tensor = this.state.activeScope.track[i];
            if (!tensor.kept && !tensorsToTrackInParentSet.has(tensor.id)) {
                tensor.dispose();
            }
        }
        const oldScope = this.state.scopeStack.pop();
        this.state.activeScope = this.state.scopeStack.length === 0 ?
            null :
            this.state.scopeStack[this.state.scopeStack.length - 1];
        // Track the current result in the parent scope.
        tensorsToTrackInParent.forEach(tensor => {
            // Only track the tensor if was allocated in the inner scope and is not
            // globally kept.
            if (!tensor.kept && tensor.scopeId === oldScope.id) {
                this.track(tensor);
            }
        });
    }
    /**
     * Returns gradients of `f` with respect to each of the `xs`. The gradients
     * returned are of the same length as `xs`, but some might be null if `f`
     * was not a function of that `x`. It also takes optional dy to multiply the
     * gradient, which defaults to `1`.
     */
    gradients(f, xs, dy, allowNoGradients = false) {
        assert(xs.length > 0, () => 'gradients() received an empty list of xs.');
        if (dy != null && dy.dtype !== 'float32') {
            throw new Error(`dy must have 'float32' dtype, but has '${dy.dtype}'`);
        }
        const y = this.scopedRun(() => this.startTape(), () => this.endTape(), () => this.tidy('forward', f));
        assert(y instanceof Tensor, () => 'The result y returned by f() must be a tensor.');
        // Filter out the nodes that don't connect x => y.
        const filteredTape = getFilteredNodesXToY(this.state.activeTape, xs, y);
        if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
            throw new Error('Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
                'that the f you passed encloses all operations that lead from x ' +
                'to y.');
        }
        return this.tidy('backward', () => {
            const accumulatedGradientMap = {};
            accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;
            // Backprop gradients through the filtered nodes.
            backpropagateGradients(accumulatedGradientMap, filteredTape, 
            // Pass the tidy function to avoid circular dep with `tape.ts`.
            f => this.tidy(f), 
            // Pass an add function to avoide a circular dep with `tape.ts`.
            add);
            const grads = xs.map(x => accumulatedGradientMap[x.id]);
            if (this.state.gradientDepth === 0) {
                // This means that we are not computing higher-order gradients
                // and can clean up the tape.
                this.state.activeTape.forEach(node => {
                    for (const tensor of node.saved) {
                        tensor.dispose();
                    }
                });
                this.state.activeTape = null;
            }
            return { value: y, grads };
        });
    }
    customGrad(f) {
        assert(isFunction(f), () => 'The f passed in customGrad(f) must be a function.');
        return (...inputs) => {
            assert(inputs.every(t => t instanceof Tensor), () => 'The args passed in customGrad(f)(x1, x2,...) must all be ' +
                'tensors');
            let res;
            const inputMap = {};
            inputs.forEach((input, i) => {
                inputMap[i] = input;
            });
            const forwardFunc = (_, save) => {
                res = f(...[...inputs, save]);
                assert(res.value instanceof Tensor, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.value` is a tensor');
                assert(isFunction(res.gradFunc), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function.');
                return res.value;
            };
            const backwardsFunc = (dy, saved) => {
                const gradRes = res.gradFunc(dy, saved);
                const grads = Array.isArray(gradRes) ? gradRes : [gradRes];
                assert(grads.length === inputs.length, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'the same number of tensors as inputs passed to f(...).');
                assert(grads.every(t => t instanceof Tensor), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'a list of only tensors.');
                const gradMap = {};
                grads.forEach((grad, i) => {
                    gradMap[i] = () => grad;
                });
                return gradMap;
            };
            return this.runKernelFunc({
                forwardFunc,
                backwardsFunc,
                inputs: inputMap,
            });
        };
    }
    readSync(dataId) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.readSync(dataId);
    }
    read(dataId) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.read(dataId);
    }
    async time(query) {
        const start = now();
        const timingInfo = await this.backend.time(query);
        timingInfo.wallMs = now() - start;
        return timingInfo;
    }
    /**
     * Tracks a Tensor in the current scope to be automatically cleaned up
     * when the current scope ends, and returns the value.
     *
     * @param result The Tensor to track in the current scope.
     */
    track(result) {
        if (this.state.activeScope != null) {
            result.scopeId = this.state.activeScope.id;
            this.state.activeScope.track.push(result);
        }
        return result;
    }
    get registeredVariables() {
        return this.state.registeredVariables;
    }
    /**
     * Resets the engine state. Removes all backends but does not remove
     * registered backend factories.
     */
    reset() {
        // Make any pending promise obsolete.
        this.pendingBackendInitId++;
        this.state.dispose();
        this.ENV.reset();
        this.state = new EngineState();
        for (const backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        this.backendName = null;
        this.backendInstance = null;
        this.pendingBackendInit = null;
    }
}
Engine.nextTensorId = 0;
Engine.nextVariableId = 0;
function ones(shape) {
    const values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
    return ENGINE.makeTensor(values, shape, 'float32');
}
function getOrMakeEngine() {
    const ns = getGlobalNamespace();
    if (ns._tfengine == null) {
        const environment = new Environment(ns);
        ns._tfengine = new Engine(environment);
    }
    setEnvironmentGlobal(ns._tfengine.ENV);
    // Tell the current tensor interface that the global engine is responsible
    // for tracking.
    setTensorTracker(() => ns._tfengine);
    return ns._tfengine;
}
const ENGINE = getOrMakeEngine();
/**
 * A implementation of the add op for use within engine and tape.
 *
 * This allows us to avoid a circular dependency between add.ts and engine.
 * It is exported to be available in tape tests.
 */
function add(a, b) {
    // We duplicate Add here to avoid a circular dependency with add.ts.
    const inputs = { a, b };
    return ENGINE.runKernel(Add, inputs);
}

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
// tslint:disable-next-line:no-any
function _isNavigatorDefined() {
    return typeof navigator !== 'undefined' && navigator != null;
}
function isMobile() {
    if (_isNavigatorDefined()) {
        // tslint:disable-next-line:no-any
        const a = navigator.userAgent || navigator.vendor || window.opera;
        // tslint:disable-next-line:max-line-length
        return /(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i
            .test(a) ||
            // tslint:disable-next-line:max-line-length
            /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i
                .test(a.substr(0, 4));
    }
    return false;
}
function isBrowser() {
    return (typeof window !== 'undefined' && window.document != null) ||
        //@ts-ignore
        (typeof WorkerGlobalScope !== 'undefined');
}

var device_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    isMobile: isMobile,
    isBrowser: isBrowser
});

const ENV$1 = env();
/**
 * This file contains environment-related flag registrations.
 */
/** Whether to enable debug mode. */
ENV$1.registerFlag('DEBUG', () => false, debugValue => {
    if (debugValue) {
        console.warn('Debugging mode is ON. The output of every math call will ' +
            'be downloaded to CPU and checked for NaNs. ' +
            'This significantly impacts performance.');
    }
});
/** Whether we are in a browser (as versus, say, node.js) environment. */
ENV$1.registerFlag('IS_BROWSER', () => isBrowser());
/** Whether we are in a browser (as versus, say, node.js) environment. */
ENV$1.registerFlag('IS_NODE', () => (typeof process !== 'undefined') &&
    (typeof process.versions !== 'undefined') &&
    (typeof process.versions.node !== 'undefined'));
/** Whether this browser is Chrome. */
ENV$1.registerFlag('IS_CHROME', () => typeof navigator !== 'undefined' && navigator != null &&
    navigator.userAgent != null && /Chrome/.test(navigator.userAgent) &&
    /Google Inc/.test(navigator.vendor));
/**
 * True when the environment is "production" where we disable safety checks
 * to gain performance.
 */
ENV$1.registerFlag('PROD', () => false);
/**
 * Whether to do sanity checks when inferring a shape from user-provided
 * values, used when creating a new tensor.
 */
ENV$1.registerFlag('TENSORLIKE_CHECK_SHAPE_CONSISTENCY', () => ENV$1.getBool('DEBUG'));
/** Whether deprecation warnings are enabled. */
ENV$1.registerFlag('DEPRECATION_WARNINGS_ENABLED', () => true);
/** True if running unit tests. */
ENV$1.registerFlag('IS_TEST', () => false);
/** Whether to check computation result for errors. */
ENV$1.registerFlag('CHECK_COMPUTATION_FOR_ERRORS', () => true);

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
function inferShape(val, dtype) {
    let firstElem = val;
    if (isTypedArray(val)) {
        return dtype === 'string' ? [] : [val.length];
    }
    if (!Array.isArray(val)) {
        return []; // Scalar.
    }
    const shape = [];
    while (Array.isArray(firstElem) ||
        isTypedArray(firstElem) && dtype !== 'string') {
        shape.push(firstElem.length);
        firstElem = firstElem[0];
    }
    if (Array.isArray(val) &&
        env().getBool('TENSORLIKE_CHECK_SHAPE_CONSISTENCY')) {
        deepAssertShapeConsistency(val, shape, []);
    }
    return shape;
}
function deepAssertShapeConsistency(val, shape, indices) {
    indices = indices || [];
    if (!(Array.isArray(val)) && !isTypedArray(val)) {
        assert(shape.length === 0, () => `Element arr[${indices.join('][')}] is a primitive, ` +
            `but should be an array/TypedArray of ${shape[0]} elements`);
        return;
    }
    assert(shape.length > 0, () => `Element arr[${indices.join('][')}] should be a primitive, ` +
        `but is an array of ${val.length} elements`);
    assert(val.length === shape[0], () => `Element arr[${indices.join('][')}] should have ${shape[0]} ` +
        `elements, but has ${val.length} elements`);
    const subShape = shape.slice(1);
    for (let i = 0; i < val.length; ++i) {
        deepAssertShapeConsistency(val[i], subShape, indices.concat(i));
    }
}
function assertDtype(expectedDtype, actualDType, argName, functionName) {
    if (expectedDtype === 'string_or_numeric') {
        return;
    }
    if (expectedDtype == null) {
        throw new Error(`Expected dtype cannot be null.`);
    }
    if (expectedDtype !== 'numeric' && expectedDtype !== actualDType ||
        expectedDtype === 'numeric' && actualDType === 'string') {
        throw new Error(`Argument '${argName}' passed to '${functionName}' must ` +
            `be ${expectedDtype} tensor, but got ${actualDType} tensor`);
    }
}
function convertToTensor(x, argName, functionName, parseAsDtype = 'numeric') {
    if (x instanceof Tensor) {
        assertDtype(parseAsDtype, x.dtype, argName, functionName);
        return x;
    }
    let inferredDtype = inferDtype(x);
    // If the user expects a bool/int/float, use that info to update the
    // inferredDtype when it is not a string.
    if (inferredDtype !== 'string' &&
        ['bool', 'int32', 'float32'].indexOf(parseAsDtype) >= 0) {
        inferredDtype = parseAsDtype;
    }
    assertDtype(parseAsDtype, inferredDtype, argName, functionName);
    if ((x == null) ||
        (!isTypedArray(x) && !Array.isArray(x) && typeof x !== 'number' &&
            typeof x !== 'boolean' && typeof x !== 'string')) {
        const type = x == null ? 'null' : x.constructor.name;
        throw new Error(`Argument '${argName}' passed to '${functionName}' must be a ` +
            `Tensor or TensorLike, but got '${type}'`);
    }
    const inferredShape = inferShape(x, inferredDtype);
    if (!isTypedArray(x) && !Array.isArray(x)) {
        x = [x];
    }
    const skipTypedArray = true;
    const values = inferredDtype !== 'string' ?
        toTypedArray(x, inferredDtype) :
        flatten(x, [], skipTypedArray);
    return ENGINE.makeTensor(values, inferredShape, inferredDtype);
}
function convertToTensorArray(arg, argName, functionName, parseAsDtype = 'numeric') {
    if (!Array.isArray(arg)) {
        throw new Error(`Argument ${argName} passed to ${functionName} must be a ` +
            '`Tensor[]` or `TensorLike[]`');
    }
    const tensors = arg;
    return tensors.map((t, i) => convertToTensor(t, `${argName}[${i}]`, functionName, parseAsDtype));
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
const OP_SCOPE_SUFFIX = '__op';
/**
 * Used for wrapping functions that perform math operations on
 * Tensors. The function will be wrapped in a named scope that cleans all
 * memory usage after the function is done.
 */
function op(f) {
    const keys = Object.keys(f);
    if (keys.length !== 1) {
        throw new Error(`Please provide an object with a single key ` +
            `(operation name) mapping to a function. Got an object with ` +
            `${keys.length} keys.`);
    }
    let opName = keys[0];
    const fn = f[opName];
    // Strip the underscore from the end of the function name.
    if (opName.endsWith('_')) {
        opName = opName.substring(0, opName.length - 1);
    }
    // add an __op suffix to distinguish ops from kernels in tf.profile
    opName = opName + OP_SCOPE_SUFFIX;
    // tslint:disable-next-line:no-any
    const f2 = (...args) => {
        ENGINE.startScope(opName);
        try {
            const result = fn(...args);
            if (isPromise(result)) {
                console.error('Cannot return a Promise inside of tidy.');
            }
            ENGINE.endScope(result);
            return result;
        }
        catch (ex) {
            ENGINE.endScope(null);
            throw ex;
        }
    };
    Object.defineProperty(f2, 'name', { value: opName, configurable: true });
    // tslint:disable-next-line:no-any
    return f2;
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
 * Converts two real numbers to a complex number.
 *
 * Given a tensor `real` representing the real part of a complex number, and a
 * tensor `imag` representing the imaginary part of a complex number, this
 * operation returns complex numbers elementwise of the form [r0, i0, r1, i1],
 * where r represents the real part and i represents the imag part.
 *
 * The input tensors real and imag must have the same shape.
 *
 * ```js
 * const real = tf.tensor1d([2.25, 3.25]);
 * const imag = tf.tensor1d([4.75, 5.75]);
 * const complex = tf.complex(real, imag);
 *
 * complex.print();
 * ```
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function complex_(real, imag) {
    const $real = convertToTensor(real, 'real', 'complex');
    const $imag = convertToTensor(imag, 'imag', 'complex');
    assertShapesMatch($real.shape, $imag.shape, `real and imag shapes, ${$real.shape} and ${$imag.shape}, ` +
        `must match in call to tf.complex().`);
    const inputs = { real: $real, imag: $imag };
    return ENGINE.runKernel(Complex, inputs);
}
const complex = op({ complex_ });

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
/** This is shared code across all tensor creation methods. */
function makeTensor(values, shape, inferredShape, dtype) {
    if (dtype == null) {
        dtype = inferDtype(values);
    }
    if (dtype === 'complex64') {
        throw new Error(`Cannot construct a complex64 tensor directly. ` +
            `Please use tf.complex(real, imag).`);
    }
    if (!isTypedArray(values) && !Array.isArray(values) &&
        typeof values !== 'number' && typeof values !== 'boolean' &&
        typeof values !== 'string') {
        throw new Error('values passed to tensor(values) must be a number/boolean/string or ' +
            'an array of numbers/booleans/strings, or a TypedArray');
    }
    if (shape != null) {
        assertNonNegativeIntegerDimensions(shape);
        const providedSize = sizeFromShape(shape);
        const inferredSize = sizeFromShape(inferredShape);
        assert(providedSize === inferredSize, () => `Based on the provided shape, [${shape}], the tensor should have ` +
            `${providedSize} values but has ${inferredSize}`);
        for (let i = 0; i < inferredShape.length; ++i) {
            const inferred = inferredShape[i];
            const flatDimsDontMatch = i === inferredShape.length - 1 ?
                inferred !== sizeFromShape(shape.slice(i)) :
                true;
            assert(inferredShape[i] === shape[i] || !flatDimsDontMatch, () => `Error creating a new Tensor. Inferred shape ` +
                `(${inferredShape}) does not match the provided ` +
                `shape (${shape}). `);
        }
    }
    if (!isTypedArray(values) && !Array.isArray(values)) {
        values = [values];
    }
    shape = shape || inferredShape;
    values = dtype !== 'string' ?
        toTypedArray(values, dtype) :
        flatten(values, [], true);
    return ENGINE.makeTensor(values, shape, dtype);
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 * Creates a `tf.Tensor` with the provided values, shape and dtype.
 *
 * ```js
 * // Pass an array of values to create a vector.
 * tf.tensor([1, 2, 3, 4]).print();
 * ```
 *
 * ```js
 * // Pass a nested array of values to make a matrix or a higher
 * // dimensional tensor.
 * tf.tensor([[1, 2], [3, 4]]).print();
 * ```
 *
 * ```js
 * // Pass a flat array and specify a shape yourself.
 * tf.tensor([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`. If the values are strings,
 *     they will be encoded as utf-8 and kept as `Uint8Array[]`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor(values, shape, dtype) {
    const inferredShape = inferShape(values, dtype);
    return makeTensor(values, shape, inferredShape, dtype);
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
/* Type definitions for exporting and importing of models. */
/**
 * A map from Tensor dtype to number of bytes per element of the Tensor.
 */
const DTYPE_VALUE_SIZE_MAP = {
    'float32': 4,
    'float16': 2,
    'int32': 4,
    'uint16': 2,
    'uint8': 1,
    'bool': 1,
    'complex64': 8
};

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
/** Number of bytes reserved for the length of the string. (32bit integer). */
const NUM_BYTES_STRING_LENGTH = 4;
/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @param group Group to which the weights belong (optional).
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
async function encodeWeights(tensors, group) {
    // TODO(adarob, cais): Support quantization.
    const specs = [];
    const dataPromises = [];
    const names = Array.isArray(tensors) ?
        tensors.map(tensor => tensor.name) :
        Object.keys(tensors);
    for (let i = 0; i < names.length; ++i) {
        const name = names[i];
        const t = Array.isArray(tensors) ? tensors[i].tensor : tensors[name];
        if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool' &&
            t.dtype !== 'string' && t.dtype !== 'complex64') {
            throw new Error(`Unsupported dtype in weight '${name}': ${t.dtype}`);
        }
        const spec = { name, shape: t.shape, dtype: t.dtype };
        if (t.dtype === 'string') {
            const utf8bytes = new Promise(async (resolve) => {
                const vals = await t.bytes();
                const totalNumBytes = vals.reduce((p, c) => p + c.length, 0) +
                    NUM_BYTES_STRING_LENGTH * vals.length;
                const bytes = new Uint8Array(totalNumBytes);
                let offset = 0;
                for (let i = 0; i < vals.length; i++) {
                    const val = vals[i];
                    const bytesOfLength = new Uint8Array(new Uint32Array([val.length]).buffer);
                    bytes.set(bytesOfLength, offset);
                    offset += NUM_BYTES_STRING_LENGTH;
                    bytes.set(val, offset);
                    offset += val.length;
                }
                resolve(bytes);
            });
            dataPromises.push(utf8bytes);
        }
        else {
            dataPromises.push(t.data());
        }
        if (group != null) {
            spec.group = group;
        }
        specs.push(spec);
    }
    const tensorValues = await Promise.all(dataPromises);
    return { data: concatenateTypedArrays(tensorValues), specs };
}
/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param buffer A flat ArrayBuffer carrying the binary values of the tensors
 *   concatenated in the order specified in `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
function decodeWeights(buffer, specs) {
    // TODO(adarob, cais): Support quantization.
    const out = {};
    let float16Decode;
    let offset = 0;
    for (const spec of specs) {
        const name = spec.name;
        const dtype = spec.dtype;
        const shape = spec.shape;
        const size = sizeFromShape(shape);
        let values;
        if ('quantization' in spec) {
            const quantization = spec.quantization;
            if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
                if (!('min' in quantization && 'scale' in quantization)) {
                    throw new Error(`Weight ${spec.name} with quantization ${quantization.dtype} ` +
                        `doesn't have corresponding metadata min and scale.`);
                }
            }
            else if (quantization.dtype === 'float16') {
                if (dtype !== 'float32') {
                    throw new Error(`Weight ${spec.name} is quantized with ${quantization.dtype} ` +
                        `which only supports weights of type float32 not ${dtype}.`);
                }
            }
            else {
                throw new Error(`Weight ${spec.name} has unknown ` +
                    `quantization dtype ${quantization.dtype}. ` +
                    `Supported quantization dtypes are: ` +
                    `'uint8', 'uint16', and 'float16'.`);
            }
            const quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
            const byteBuffer = buffer.slice(offset, offset + size * quantizationSizeFactor);
            const quantizedArray = (quantization.dtype === 'uint8') ?
                new Uint8Array(byteBuffer) :
                new Uint16Array(byteBuffer);
            if (dtype === 'float32') {
                if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
                    values = new Float32Array(quantizedArray.length);
                    for (let i = 0; i < quantizedArray.length; i++) {
                        const v = quantizedArray[i];
                        values[i] = v * quantization.scale + quantization.min;
                    }
                }
                else if (quantization.dtype === 'float16') {
                    if (float16Decode === undefined) {
                        float16Decode = getFloat16Decoder();
                    }
                    values = float16Decode(quantizedArray);
                }
                else {
                    throw new Error(`Unsupported quantization type ${quantization.dtype} ` +
                        `for weight type float32.`);
                }
            }
            else if (dtype === 'int32') {
                if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
                    throw new Error(`Unsupported quantization type ${quantization.dtype} ` +
                        `for weight type int32.`);
                }
                values = new Int32Array(quantizedArray.length);
                for (let i = 0; i < quantizedArray.length; i++) {
                    const v = quantizedArray[i];
                    values[i] = Math.round(v * quantization.scale + quantization.min);
                }
            }
            else {
                throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
            }
            offset += size * quantizationSizeFactor;
        }
        else if (dtype === 'string') {
            const size = sizeFromShape(spec.shape);
            values = [];
            for (let i = 0; i < size; i++) {
                const byteLength = new Uint32Array(buffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
                offset += NUM_BYTES_STRING_LENGTH;
                const bytes = new Uint8Array(buffer.slice(offset, offset + byteLength));
                values.push(bytes);
                offset += byteLength;
            }
        }
        else {
            const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
            const byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);
            if (dtype === 'float32') {
                values = new Float32Array(byteBuffer);
            }
            else if (dtype === 'int32') {
                values = new Int32Array(byteBuffer);
            }
            else if (dtype === 'bool') {
                values = new Uint8Array(byteBuffer);
            }
            else if (dtype === 'complex64') {
                values = new Float32Array(byteBuffer);
                const real = new Float32Array(values.length / 2);
                const image = new Float32Array(values.length / 2);
                for (let i = 0; i < real.length; i++) {
                    real[i] = values[i * 2];
                    image[i] = values[i * 2 + 1];
                }
                const realTensor = tensor(real, shape, 'float32');
                const imageTensor = tensor(image, shape, 'float32');
                out[name] = complex(realTensor, imageTensor);
                realTensor.dispose();
                imageTensor.dispose();
            }
            else {
                throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
            }
            offset += size * dtypeFactor;
        }
        if (dtype !== 'complex64') {
            out[name] = tensor(values, shape, dtype);
        }
    }
    return out;
}
/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
function concatenateTypedArrays(xs) {
    // TODO(adarob, cais): Support quantization.
    if (xs === null) {
        throw new Error(`Invalid input value: ${JSON.stringify(xs)}`);
    }
    let totalByteLength = 0;
    // `normalizedXs` is here for this reason: a `TypedArray`'s `buffer'
    // can have a different byte length from that of the `TypedArray` itself,
    // for example, when the `TypedArray` is created from an offset in an
    // `ArrayBuffer`. `normliazedXs` holds `TypedArray`s whose `buffer`s match
    // the `TypedArray` in byte length. If an element of `xs` does not show
    // this property, a new `TypedArray` that satisfy this property will be
    // constructed and pushed into `normalizedXs`.
    const normalizedXs = [];
    xs.forEach((x) => {
        totalByteLength += x.byteLength;
        // tslint:disable:no-any
        normalizedXs.push(x.byteLength === x.buffer.byteLength ? x :
            new x.constructor(x));
        if (!(x instanceof Float32Array || x instanceof Int32Array ||
            x instanceof Uint8Array)) {
            throw new Error(`Unsupported TypedArray subtype: ${x.constructor.name}`);
        }
        // tslint:enable:no-any
    });
    const y = new Uint8Array(totalByteLength);
    let offset = 0;
    normalizedXs.forEach((x) => {
        y.set(new Uint8Array(x.buffer), offset);
        offset += x.byteLength;
    });
    return y.buffer;
}
// Use Buffer on Node.js instead of Blob/atob/btoa
const useNodeBuffer = typeof Buffer !== 'undefined' &&
    (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
        typeof btoa === 'undefined');
/**
 * Calculate the byte length of a JavaScript string.
 *
 * Note that a JavaScript string can contain wide characters, therefore the
 * length of the string is not necessarily equal to the byte length.
 *
 * @param str Input string.
 * @returns Byte length.
 */
function stringByteLength(str) {
    if (useNodeBuffer) {
        return Buffer.byteLength(str);
    }
    return new Blob([str]).size;
}
/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
function arrayBufferToBase64String(buffer) {
    if (useNodeBuffer) {
        return Buffer.from(buffer).toString('base64');
    }
    const buf = new Uint8Array(buffer);
    let s = '';
    for (let i = 0, l = buf.length; i < l; i++) {
        s += String.fromCharCode(buf[i]);
    }
    return btoa(s);
}
/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
function base64StringToArrayBuffer(str) {
    if (useNodeBuffer) {
        const buf = Buffer.from(str, 'base64');
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    const s = atob(str);
    const buffer = new Uint8Array(s.length);
    for (let i = 0; i < s.length; ++i) {
        buffer.set([s.charCodeAt(i)], i);
    }
    return buffer.buffer;
}
/**
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers A number of array buffers to concatenate.
 * @returns Result of concatenating `buffers` in order.
 */
function concatenateArrayBuffers(buffers) {
    if (buffers.length === 1) {
        return buffers[0];
    }
    let totalByteLength = 0;
    buffers.forEach((buffer) => {
        totalByteLength += buffer.byteLength;
    });
    const temp = new Uint8Array(totalByteLength);
    let offset = 0;
    buffers.forEach((buffer) => {
        temp.set(new Uint8Array(buffer), offset);
        offset += buffer.byteLength;
    });
    return temp.buffer;
}
/**
 * Get the basename of a path.
 *
 * Behaves in a way analogous to Linux's basename command.
 *
 * @param path
 */
function basename(path) {
    const SEPARATOR = '/';
    path = path.trim();
    while (path.endsWith(SEPARATOR)) {
        path = path.slice(0, path.length - 1);
    }
    const items = path.split(SEPARATOR);
    return items[items.length - 1];
}
/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
function getModelArtifactsInfoForJSON(modelArtifacts) {
    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
        throw new Error('Expected JSON model topology, received ArrayBuffer.');
    }
    return {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: modelArtifacts.modelTopology == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
        weightSpecsBytes: modelArtifacts.weightSpecs == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
        weightDataBytes: modelArtifacts.weightData == null ?
            0 :
            modelArtifacts.weightData.byteLength,
    };
}
/**
 * Computes mantisa table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 2048 mantissa lookup values.
 */
function computeFloat16MantisaTable() {
    const convertMantissa = (i) => {
        let m = i << 13;
        let e = 0;
        while ((m & 0x00800000) === 0) {
            e -= 0x00800000;
            m <<= 1;
        }
        m &= ~0x00800000;
        e += 0x38800000;
        return m | e;
    };
    const mantisaTable = new Uint32Array(2048);
    mantisaTable[0] = 0;
    for (let i = 1; i < 1024; i++) {
        mantisaTable[i] = convertMantissa(i);
    }
    for (let i = 1024; i < 2048; i++) {
        mantisaTable[i] = 0x38000000 + ((i - 1024) << 13);
    }
    return mantisaTable;
}
/**
 * Computes exponent table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 64 exponent lookup values.
 */
function computeFloat16ExponentTable() {
    const exponentTable = new Uint32Array(64);
    exponentTable[0] = 0;
    exponentTable[31] = 0x47800000;
    exponentTable[32] = 0x80000000;
    exponentTable[63] = 0xc7800000;
    for (let i = 1; i < 31; i++) {
        exponentTable[i] = i << 23;
    }
    for (let i = 33; i < 63; i++) {
        exponentTable[i] = 0x80000000 + ((i - 32) << 23);
    }
    return exponentTable;
}
/**
 * Computes offset table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 6d offset values.
 */
function computeFloat16OffsetTable() {
    const offsetTable = new Uint32Array(64);
    for (let i = 0; i < 64; i++) {
        offsetTable[i] = 1024;
    }
    offsetTable[0] = offsetTable[32] = 0;
    return offsetTable;
}
/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
function getFloat16Decoder() {
    // Algorithm is based off of
    // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
    // Cache lookup tables
    const mantisaTable = computeFloat16MantisaTable();
    const exponentTable = computeFloat16ExponentTable();
    const offsetTable = computeFloat16OffsetTable();
    return (quantizedArray) => {
        const buffer = new ArrayBuffer(4 * quantizedArray.length);
        const bufferUint32View = new Uint32Array(buffer);
        for (let index = 0; index < quantizedArray.length; index++) {
            const float16Bits = quantizedArray[index];
            const float32Bits = mantisaTable[offsetTable[float16Bits >> 10] + (float16Bits & 0x3ff)] +
                exponentTable[float16Bits >> 10];
            bufferUint32View[index] = float32Bits;
        }
        return new Float32Array(buffer);
    };
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
class IORouterRegistry {
    constructor() {
        this.saveRouters = [];
        this.loadRouters = [];
    }
    static getInstance() {
        if (IORouterRegistry.instance == null) {
            IORouterRegistry.instance = new IORouterRegistry();
        }
        return IORouterRegistry.instance;
    }
    /**
     * Register a save-handler router.
     *
     * @param saveRouter A function that maps a URL-like string onto an instance
     * of `IOHandler` with the `save` method defined or `null`.
     */
    static registerSaveRouter(saveRouter) {
        IORouterRegistry.getInstance().saveRouters.push(saveRouter);
    }
    /**
     * Register a load-handler router.
     *
     * @param loadRouter A function that maps a URL-like string onto an instance
     * of `IOHandler` with the `load` method defined or `null`.
     */
    static registerLoadRouter(loadRouter) {
        IORouterRegistry.getInstance().loadRouters.push(loadRouter);
    }
    /**
     * Look up IOHandler for saving, given a URL-like string.
     *
     * @param url
     * @returns If only one match is found, an instance of IOHandler with the
     * `save` method defined. If no match is found, `null`.
     * @throws Error, if more than one match is found.
     */
    static getSaveHandlers(url) {
        return IORouterRegistry.getHandlers(url, 'save');
    }
    /**
     * Look up IOHandler for loading, given a URL-like string.
     *
     * @param url
     * @param loadOptions Optional, custom load options.
     * @returns All valid handlers for `url`, given the currently registered
     *   handler routers.
     */
    static getLoadHandlers(url, loadOptions) {
        return IORouterRegistry.getHandlers(url, 'load', loadOptions);
    }
    static getHandlers(url, handlerType, loadOptions) {
        const validHandlers = [];
        const routers = handlerType === 'load' ?
            IORouterRegistry.getInstance().loadRouters :
            IORouterRegistry.getInstance().saveRouters;
        routers.forEach(router => {
            const handler = router(url, loadOptions);
            if (handler !== null) {
                validHandlers.push(handler);
            }
        });
        return validHandlers;
    }
}
const registerSaveRouter = (loudRouter) => IORouterRegistry.registerSaveRouter(loudRouter);
const registerLoadRouter = (loudRouter) => IORouterRegistry.registerLoadRouter(loudRouter);
const getSaveHandlers = (url) => IORouterRegistry.getSaveHandlers(url);
const getLoadHandlers = (url, loadOptions) => IORouterRegistry.getLoadHandlers(url, loadOptions);

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
const DATABASE_NAME = 'tensorflowjs';
const DATABASE_VERSION = 1;
// Model data and ModelArtifactsInfo (metadata) are stored in two separate
// stores for efficient access of the list of stored models and their metadata.
// 1. The object store for model data: topology, weights and weight manifests.
const MODEL_STORE_NAME = 'models_store';
// 2. The object store for ModelArtifactsInfo, including meta-information such
//    as the type of topology (JSON vs binary), byte size of the topology, byte
//    size of the weights, etc.
const INFO_STORE_NAME = 'model_info_store';
function getIndexedDBFactory() {
    if (!env().getBool('IS_BROWSER')) {
        // TODO(cais): Add more info about what IOHandler subtypes are available.
        //   Maybe point to a doc page on the web and/or automatically determine
        //   the available IOHandlers and print them in the error message.
        throw new Error('Failed to obtain IndexedDB factory because the current environment' +
            'is not a web browser.');
    }
    // tslint:disable-next-line:no-any
    const theWindow = typeof window === 'undefined' ? self : window;
    const factory = theWindow.indexedDB || theWindow.mozIndexedDB ||
        theWindow.webkitIndexedDB || theWindow.msIndexedDB ||
        theWindow.shimIndexedDB;
    if (factory == null) {
        throw new Error('The current browser does not appear to support IndexedDB.');
    }
    return factory;
}
function setUpDatabase(openRequest) {
    const db = openRequest.result;
    db.createObjectStore(MODEL_STORE_NAME, { keyPath: 'modelPath' });
    db.createObjectStore(INFO_STORE_NAME, { keyPath: 'modelPath' });
}
/**
 * IOHandler subclass: Browser IndexedDB.
 *
 * See the doc string of `browserIndexedDB` for more details.
 */
class BrowserIndexedDB {
    constructor(modelPath) {
        this.indexedDB = getIndexedDBFactory();
        if (modelPath == null || !modelPath) {
            throw new Error('For IndexedDB, modelPath must not be null, undefined or empty.');
        }
        this.modelPath = modelPath;
    }
    async save(modelArtifacts) {
        // TODO(cais): Support saving GraphDef models.
        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
            throw new Error('BrowserLocalStorage.save() does not support saving model topology ' +
                'in binary formats yet.');
        }
        return this.databaseAction(this.modelPath, modelArtifacts);
    }
    async load() {
        return this.databaseAction(this.modelPath);
    }
    /**
     * Perform database action to put model artifacts into or read model artifacts
     * from IndexedDB object store.
     *
     * Whether the action is put or get depends on whether `modelArtifacts` is
     * specified. If it is specified, the action will be put; otherwise the action
     * will be get.
     *
     * @param modelPath A unique string path for the model.
     * @param modelArtifacts If specified, it will be the model artifacts to be
     *   stored in IndexedDB.
     * @returns A `Promise` of `SaveResult`, if the action is put, or a `Promise`
     *   of `ModelArtifacts`, if the action is get.
     */
    databaseAction(modelPath, modelArtifacts) {
        return new Promise((resolve, reject) => {
            const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
            openRequest.onupgradeneeded = () => setUpDatabase(openRequest);
            openRequest.onsuccess = () => {
                const db = openRequest.result;
                if (modelArtifacts == null) {
                    // Read model out from object store.
                    const modelTx = db.transaction(MODEL_STORE_NAME, 'readonly');
                    const modelStore = modelTx.objectStore(MODEL_STORE_NAME);
                    const getRequest = modelStore.get(this.modelPath);
                    getRequest.onsuccess = () => {
                        if (getRequest.result == null) {
                            db.close();
                            return reject(new Error(`Cannot find model with path '${this.modelPath}' ` +
                                `in IndexedDB.`));
                        }
                        else {
                            resolve(getRequest.result.modelArtifacts);
                        }
                    };
                    getRequest.onerror = error => {
                        db.close();
                        return reject(getRequest.error);
                    };
                    modelTx.oncomplete = () => db.close();
                }
                else {
                    // Put model into object store.
                    const modelArtifactsInfo = getModelArtifactsInfoForJSON(modelArtifacts);
                    // First, put ModelArtifactsInfo into info store.
                    const infoTx = db.transaction(INFO_STORE_NAME, 'readwrite');
                    let infoStore = infoTx.objectStore(INFO_STORE_NAME);
                    const putInfoRequest = infoStore.put({ modelPath: this.modelPath, modelArtifactsInfo });
                    let modelTx;
                    putInfoRequest.onsuccess = () => {
                        // Second, put model data into model store.
                        modelTx = db.transaction(MODEL_STORE_NAME, 'readwrite');
                        const modelStore = modelTx.objectStore(MODEL_STORE_NAME);
                        const putModelRequest = modelStore.put({
                            modelPath: this.modelPath,
                            modelArtifacts,
                            modelArtifactsInfo
                        });
                        putModelRequest.onsuccess = () => resolve({ modelArtifactsInfo });
                        putModelRequest.onerror = error => {
                            // If the put-model request fails, roll back the info entry as
                            // well.
                            infoStore = infoTx.objectStore(INFO_STORE_NAME);
                            const deleteInfoRequest = infoStore.delete(this.modelPath);
                            deleteInfoRequest.onsuccess = () => {
                                db.close();
                                return reject(putModelRequest.error);
                            };
                            deleteInfoRequest.onerror = error => {
                                db.close();
                                return reject(putModelRequest.error);
                            };
                        };
                    };
                    putInfoRequest.onerror = error => {
                        db.close();
                        return reject(putInfoRequest.error);
                    };
                    infoTx.oncomplete = () => {
                        if (modelTx == null) {
                            db.close();
                        }
                        else {
                            modelTx.oncomplete = () => db.close();
                        }
                    };
                }
            };
            openRequest.onerror = error => reject(openRequest.error);
        });
    }
}
BrowserIndexedDB.URL_SCHEME = 'indexeddb://';
const indexedDBRouter = (url) => {
    if (!env().getBool('IS_BROWSER')) {
        return null;
    }
    else {
        if (!Array.isArray(url) && url.startsWith(BrowserIndexedDB.URL_SCHEME)) {
            return browserIndexedDB(url.slice(BrowserIndexedDB.URL_SCHEME.length));
        }
        else {
            return null;
        }
    }
};
IORouterRegistry.registerSaveRouter(indexedDBRouter);
IORouterRegistry.registerLoadRouter(indexedDBRouter);
/**
 * Creates a browser IndexedDB IOHandler for saving and loading models.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(
 *     tf.layers.dense({units: 1, inputShape: [100], activation: 'sigmoid'}));
 *
 * const saveResult = await model.save('indexeddb://MyModel'));
 * console.log(saveResult);
 * ```
 *
 * @param modelPath A unique identifier for the model to be saved. Must be a
 *   non-empty string.
 * @returns An instance of `BrowserIndexedDB` (sublcass of `IOHandler`),
 *   which can be used with, e.g., `tf.Model.save`.
 */
function browserIndexedDB(modelPath) {
    return new BrowserIndexedDB(modelPath);
}
function maybeStripScheme(key) {
    return key.startsWith(BrowserIndexedDB.URL_SCHEME) ?
        key.slice(BrowserIndexedDB.URL_SCHEME.length) :
        key;
}
class BrowserIndexedDBManager {
    constructor() {
        this.indexedDB = getIndexedDBFactory();
    }
    async listModels() {
        return new Promise((resolve, reject) => {
            const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
            openRequest.onupgradeneeded = () => setUpDatabase(openRequest);
            openRequest.onsuccess = () => {
                const db = openRequest.result;
                const tx = db.transaction(INFO_STORE_NAME, 'readonly');
                const store = tx.objectStore(INFO_STORE_NAME);
                // tslint:disable:max-line-length
                // Need to cast `store` as `any` here because TypeScript's DOM
                // library does not have the `getAll()` method even though the
                // method is supported in the latest version of most mainstream
                // browsers:
                // https://developer.mozilla.org/en-US/docs/Web/API/IDBObjectStore/getAll
                // tslint:enable:max-line-length
                // tslint:disable-next-line:no-any
                const getAllInfoRequest = store.getAll();
                getAllInfoRequest.onsuccess = () => {
                    const out = {};
                    for (const item of getAllInfoRequest.result) {
                        out[item.modelPath] = item.modelArtifactsInfo;
                    }
                    resolve(out);
                };
                getAllInfoRequest.onerror = error => {
                    db.close();
                    return reject(getAllInfoRequest.error);
                };
                tx.oncomplete = () => db.close();
            };
            openRequest.onerror = error => reject(openRequest.error);
        });
    }
    async removeModel(path) {
        path = maybeStripScheme(path);
        return new Promise((resolve, reject) => {
            const openRequest = this.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
            openRequest.onupgradeneeded = () => setUpDatabase(openRequest);
            openRequest.onsuccess = () => {
                const db = openRequest.result;
                const infoTx = db.transaction(INFO_STORE_NAME, 'readwrite');
                const infoStore = infoTx.objectStore(INFO_STORE_NAME);
                const getInfoRequest = infoStore.get(path);
                let modelTx;
                getInfoRequest.onsuccess = () => {
                    if (getInfoRequest.result == null) {
                        db.close();
                        return reject(new Error(`Cannot find model with path '${path}' ` +
                            `in IndexedDB.`));
                    }
                    else {
                        // First, delete the entry in the info store.
                        const deleteInfoRequest = infoStore.delete(path);
                        const deleteModelData = () => {
                            // Second, delete the entry in the model store.
                            modelTx = db.transaction(MODEL_STORE_NAME, 'readwrite');
                            const modelStore = modelTx.objectStore(MODEL_STORE_NAME);
                            const deleteModelRequest = modelStore.delete(path);
                            deleteModelRequest.onsuccess = () => resolve(getInfoRequest.result.modelArtifactsInfo);
                            deleteModelRequest.onerror = error => reject(getInfoRequest.error);
                        };
                        // Proceed with deleting model data regardless of whether deletion
                        // of info data succeeds or not.
                        deleteInfoRequest.onsuccess = deleteModelData;
                        deleteInfoRequest.onerror = error => {
                            deleteModelData();
                            db.close();
                            return reject(getInfoRequest.error);
                        };
                    }
                };
                getInfoRequest.onerror = error => {
                    db.close();
                    return reject(getInfoRequest.error);
                };
                infoTx.oncomplete = () => {
                    if (modelTx == null) {
                        db.close();
                    }
                    else {
                        modelTx.oncomplete = () => db.close();
                    }
                };
            };
            openRequest.onerror = error => reject(openRequest.error);
        });
    }
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
const PATH_SEPARATOR = '/';
const PATH_PREFIX = 'tensorflowjs_models';
const INFO_SUFFIX = 'info';
const MODEL_TOPOLOGY_SUFFIX = 'model_topology';
const WEIGHT_SPECS_SUFFIX = 'weight_specs';
const WEIGHT_DATA_SUFFIX = 'weight_data';
const MODEL_METADATA_SUFFIX = 'model_metadata';
function getModelKeys(path) {
    return {
        info: [PATH_PREFIX, path, INFO_SUFFIX].join(PATH_SEPARATOR),
        topology: [PATH_PREFIX, path, MODEL_TOPOLOGY_SUFFIX].join(PATH_SEPARATOR),
        weightSpecs: [PATH_PREFIX, path, WEIGHT_SPECS_SUFFIX].join(PATH_SEPARATOR),
        weightData: [PATH_PREFIX, path, WEIGHT_DATA_SUFFIX].join(PATH_SEPARATOR),
        modelMetadata: [PATH_PREFIX, path, MODEL_METADATA_SUFFIX].join(PATH_SEPARATOR)
    };
}
/**
 * Get model path from a local-storage key.
 *
 * E.g., 'tensorflowjs_models/my/model/1/info' --> 'my/model/1'
 *
 * @param key
 */
function getModelPathFromKey(key) {
    const items = key.split(PATH_SEPARATOR);
    if (items.length < 3) {
        throw new Error(`Invalid key format: ${key}`);
    }
    return items.slice(1, items.length - 1).join(PATH_SEPARATOR);
}
function maybeStripScheme$1(key) {
    return key.startsWith(BrowserLocalStorage.URL_SCHEME) ?
        key.slice(BrowserLocalStorage.URL_SCHEME.length) :
        key;
}
/**
 * IOHandler subclass: Browser Local Storage.
 *
 * See the doc string to `browserLocalStorage` for more details.
 */
class BrowserLocalStorage {
    constructor(modelPath) {
        if (!env().getBool('IS_BROWSER') || typeof window === 'undefined' ||
            typeof window.localStorage === 'undefined') {
            // TODO(cais): Add more info about what IOHandler subtypes are
            // available.
            //   Maybe point to a doc page on the web and/or automatically determine
            //   the available IOHandlers and print them in the error message.
            throw new Error('The current environment does not support local storage.');
        }
        this.LS = window.localStorage;
        if (modelPath == null || !modelPath) {
            throw new Error('For local storage, modelPath must not be null, undefined or empty.');
        }
        this.modelPath = modelPath;
        this.keys = getModelKeys(this.modelPath);
    }
    /**
     * Save model artifacts to browser local storage.
     *
     * See the documentation to `browserLocalStorage` for details on the saved
     * artifacts.
     *
     * @param modelArtifacts The model artifacts to be stored.
     * @returns An instance of SaveResult.
     */
    async save(modelArtifacts) {
        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
            throw new Error('BrowserLocalStorage.save() does not support saving model topology ' +
                'in binary formats yet.');
        }
        else {
            const topology = JSON.stringify(modelArtifacts.modelTopology);
            const weightSpecs = JSON.stringify(modelArtifacts.weightSpecs);
            const modelArtifactsInfo = getModelArtifactsInfoForJSON(modelArtifacts);
            try {
                this.LS.setItem(this.keys.info, JSON.stringify(modelArtifactsInfo));
                this.LS.setItem(this.keys.topology, topology);
                this.LS.setItem(this.keys.weightSpecs, weightSpecs);
                this.LS.setItem(this.keys.weightData, arrayBufferToBase64String(modelArtifacts.weightData));
                const result = {
                    format: modelArtifacts.format,
                    generatedBy: modelArtifacts.generatedBy,
                    convertedBy: modelArtifacts.convertedBy
                };
                if (modelArtifacts.signature != null) {
                    result.signature = modelArtifacts.signature;
                }
                if (modelArtifacts.userDefinedMetadata != null) {
                    result.userDefinedMetadata = modelArtifacts.userDefinedMetadata;
                }
                if (modelArtifacts.modelInitializer != null) {
                    result.modelInitializer = modelArtifacts.modelInitializer;
                }
                this.LS.setItem(this.keys.modelMetadata, JSON.stringify(result));
                return { modelArtifactsInfo };
            }
            catch (err) {
                // If saving failed, clean up all items saved so far.
                this.LS.removeItem(this.keys.info);
                this.LS.removeItem(this.keys.topology);
                this.LS.removeItem(this.keys.weightSpecs);
                this.LS.removeItem(this.keys.weightData);
                this.LS.removeItem(this.keys.modelMetadata);
                throw new Error(`Failed to save model '${this.modelPath}' to local storage: ` +
                    `size quota being exceeded is a possible cause of this failure: ` +
                    `modelTopologyBytes=${modelArtifactsInfo.modelTopologyBytes}, ` +
                    `weightSpecsBytes=${modelArtifactsInfo.weightSpecsBytes}, ` +
                    `weightDataBytes=${modelArtifactsInfo.weightDataBytes}.`);
            }
        }
    }
    /**
     * Load a model from local storage.
     *
     * See the documentation to `browserLocalStorage` for details on the saved
     * artifacts.
     *
     * @returns The loaded model (if loading succeeds).
     */
    async load() {
        const info = JSON.parse(this.LS.getItem(this.keys.info));
        if (info == null) {
            throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);
        }
        if (info.modelTopologyType !== 'JSON') {
            throw new Error('BrowserLocalStorage does not support loading non-JSON model ' +
                'topology yet.');
        }
        const out = {};
        // Load topology.
        const topology = JSON.parse(this.LS.getItem(this.keys.topology));
        if (topology == null) {
            throw new Error(`In local storage, the topology of model '${this.modelPath}' ` +
                `is missing.`);
        }
        out.modelTopology = topology;
        // Load weight specs.
        const weightSpecs = JSON.parse(this.LS.getItem(this.keys.weightSpecs));
        if (weightSpecs == null) {
            throw new Error(`In local storage, the weight specs of model '${this.modelPath}' ` +
                `are missing.`);
        }
        out.weightSpecs = weightSpecs;
        // Load meta-data fields.
        const metadataString = this.LS.getItem(this.keys.modelMetadata);
        if (metadataString != null) {
            const metadata = JSON.parse(metadataString);
            out.format = metadata['format'];
            out.generatedBy = metadata['generatedBy'];
            out.convertedBy = metadata['convertedBy'];
            if (metadata['signature'] != null) {
                out.signature = metadata['signature'];
            }
            if (metadata['userDefinedMetadata'] != null) {
                out.userDefinedMetadata = metadata['userDefinedMetadata'];
            }
            if (metadata['modelInitializer'] != null) {
                out.modelInitializer = metadata['modelInitializer'];
            }
        }
        // Load weight data.
        const weightDataBase64 = this.LS.getItem(this.keys.weightData);
        if (weightDataBase64 == null) {
            throw new Error(`In local storage, the binary weight values of model ` +
                `'${this.modelPath}' are missing.`);
        }
        out.weightData = base64StringToArrayBuffer(weightDataBase64);
        return out;
    }
}
BrowserLocalStorage.URL_SCHEME = 'localstorage://';
const localStorageRouter = (url) => {
    if (!env().getBool('IS_BROWSER')) {
        return null;
    }
    else {
        if (!Array.isArray(url) && url.startsWith(BrowserLocalStorage.URL_SCHEME)) {
            return browserLocalStorage(url.slice(BrowserLocalStorage.URL_SCHEME.length));
        }
        else {
            return null;
        }
    }
};
IORouterRegistry.registerSaveRouter(localStorageRouter);
IORouterRegistry.registerLoadRouter(localStorageRouter);
/**
 * Factory function for local storage IOHandler.
 *
 * This `IOHandler` supports both `save` and `load`.
 *
 * For each model's saved artifacts, four items are saved to local storage.
 *   - `${PATH_SEPARATOR}/${modelPath}/info`: Contains meta-info about the
 *     model, such as date saved, type of the topology, size in bytes, etc.
 *   - `${PATH_SEPARATOR}/${modelPath}/topology`: Model topology. For Keras-
 *     style models, this is a stringized JSON.
 *   - `${PATH_SEPARATOR}/${modelPath}/weight_specs`: Weight specs of the
 *     model, can be used to decode the saved binary weight values (see
 *     item below).
 *   - `${PATH_SEPARATOR}/${modelPath}/weight_data`: Concatenated binary
 *     weight values, stored as a base64-encoded string.
 *
 * Saving may throw an `Error` if the total size of the artifacts exceed the
 * browser-specific quota.
 *
 * @param modelPath A unique identifier for the model to be saved. Must be a
 *   non-empty string.
 * @returns An instance of `IOHandler`, which can be used with, e.g.,
 *   `tf.Model.save`.
 */
function browserLocalStorage(modelPath) {
    return new BrowserLocalStorage(modelPath);
}
class BrowserLocalStorageManager {
    constructor() {
        assert(env().getBool('IS_BROWSER'), () => 'Current environment is not a web browser');
        assert(typeof window === 'undefined' ||
            typeof window.localStorage !== 'undefined', () => 'Current browser does not appear to support localStorage');
        this.LS = window.localStorage;
    }
    async listModels() {
        const out = {};
        const prefix = PATH_PREFIX + PATH_SEPARATOR;
        const suffix = PATH_SEPARATOR + INFO_SUFFIX;
        for (let i = 0; i < this.LS.length; ++i) {
            const key = this.LS.key(i);
            if (key.startsWith(prefix) && key.endsWith(suffix)) {
                const modelPath = getModelPathFromKey(key);
                out[modelPath] = JSON.parse(this.LS.getItem(key));
            }
        }
        return out;
    }
    async removeModel(path) {
        path = maybeStripScheme$1(path);
        const keys = getModelKeys(path);
        if (this.LS.getItem(keys.info) == null) {
            throw new Error(`Cannot find model at path '${path}'`);
        }
        const info = JSON.parse(this.LS.getItem(keys.info));
        this.LS.removeItem(keys.info);
        this.LS.removeItem(keys.topology);
        this.LS.removeItem(keys.weightSpecs);
        this.LS.removeItem(keys.weightData);
        return info;
    }
}

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
const URL_SCHEME_SUFFIX = '://';
class ModelStoreManagerRegistry {
    constructor() {
        this.managers = {};
    }
    static getInstance() {
        if (ModelStoreManagerRegistry.instance == null) {
            ModelStoreManagerRegistry.instance = new ModelStoreManagerRegistry();
        }
        return ModelStoreManagerRegistry.instance;
    }
    /**
     * Register a save-handler router.
     *
     * @param saveRouter A function that maps a URL-like string onto an instance
     * of `IOHandler` with the `save` method defined or `null`.
     */
    static registerManager(scheme, manager) {
        assert(scheme != null, () => 'scheme must not be undefined or null.');
        if (scheme.endsWith(URL_SCHEME_SUFFIX)) {
            scheme = scheme.slice(0, scheme.indexOf(URL_SCHEME_SUFFIX));
        }
        assert(scheme.length > 0, () => 'scheme must not be an empty string.');
        const registry = ModelStoreManagerRegistry.getInstance();
        assert(registry.managers[scheme] == null, () => `A model store manager is already registered for scheme '${scheme}'.`);
        registry.managers[scheme] = manager;
    }
    static getManager(scheme) {
        const manager = this.getInstance().managers[scheme];
        if (manager == null) {
            throw new Error(`Cannot find model manager for scheme '${scheme}'`);
        }
        return manager;
    }
    static getSchemes() {
        return Object.keys(this.getInstance().managers);
    }
}
/**
 * Helper method for parsing a URL string into a scheme and a path.
 *
 * @param url E.g., 'localstorage://my-model'
 * @returns A dictionary with two fields: scheme and path.
 *   Scheme: e.g., 'localstorage' in the example above.
 *   Path: e.g., 'my-model' in the example above.
 */
function parseURL(url) {
    if (url.indexOf(URL_SCHEME_SUFFIX) === -1) {
        throw new Error(`The url string provided does not contain a scheme. ` +
            `Supported schemes are: ` +
            `${ModelStoreManagerRegistry.getSchemes().join(',')}`);
    }
    return {
        scheme: url.split(URL_SCHEME_SUFFIX)[0],
        path: url.split(URL_SCHEME_SUFFIX)[1],
    };
}
async function cloneModelInternal(sourceURL, destURL, deleteSource = false) {
    assert(sourceURL !== destURL, () => `Old path and new path are the same: '${sourceURL}'`);
    const loadHandlers = IORouterRegistry.getLoadHandlers(sourceURL);
    assert(loadHandlers.length > 0, () => `Copying failed because no load handler is found for source URL ${sourceURL}.`);
    assert(loadHandlers.length < 2, () => `Copying failed because more than one (${loadHandlers.length}) ` +
        `load handlers for source URL ${sourceURL}.`);
    const loadHandler = loadHandlers[0];
    const saveHandlers = IORouterRegistry.getSaveHandlers(destURL);
    assert(saveHandlers.length > 0, () => `Copying failed because no save handler is found for destination ` +
        `URL ${destURL}.`);
    assert(saveHandlers.length < 2, () => `Copying failed because more than one (${loadHandlers.length}) ` +
        `save handlers for destination URL ${destURL}.`);
    const saveHandler = saveHandlers[0];
    const sourceScheme = parseURL(sourceURL).scheme;
    const sourcePath = parseURL(sourceURL).path;
    const sameMedium = sourceScheme === parseURL(sourceURL).scheme;
    const modelArtifacts = await loadHandler.load();
    // If moving within the same storage medium, remove the old model as soon as
    // the loading is done. Without doing this, it is possible that the combined
    // size of the two models will cause the cloning to fail.
    if (deleteSource && sameMedium) {
        await ModelStoreManagerRegistry.getManager(sourceScheme)
            .removeModel(sourcePath);
    }
    const saveResult = await saveHandler.save(modelArtifacts);
    // If moving between mediums, the deletion is done after the save succeeds.
    // This guards against the case in which saving to the destination medium
    // fails.
    if (deleteSource && !sameMedium) {
        await ModelStoreManagerRegistry.getManager(sourceScheme)
            .removeModel(sourcePath);
    }
    return saveResult.modelArtifactsInfo;
}
/**
 * List all models stored in registered storage mediums.
 *
 * For a web browser environment, the registered mediums are Local Storage and
 * IndexedDB.
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Delete the model.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 * ```
 *
 * @returns A `Promise` of a dictionary mapping URLs of existing models to
 * their model artifacts info. URLs include medium-specific schemes, e.g.,
 *   'indexeddb://my/model/1'. Model artifacts info include type of the
 * model's topology, byte sizes of the topology, weights, etc.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function listModels() {
    const schemes = ModelStoreManagerRegistry.getSchemes();
    const out = {};
    for (const scheme of schemes) {
        const schemeOut = await ModelStoreManagerRegistry.getManager(scheme).listModels();
        for (const path in schemeOut) {
            const url = scheme + URL_SCHEME_SUFFIX + path;
            out[url] = schemeOut[path];
        }
    }
    return out;
}
/**
 * Remove a model specified by URL from a reigstered storage medium.
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Delete the model.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 * ```
 *
 * @param url A URL to a stored model, with a scheme prefix, e.g.,
 *   'localstorage://my-model-1', 'indexeddb://my/model/2'.
 * @returns ModelArtifactsInfo of the deleted model (if and only if deletion
 *   is successful).
 * @throws Error if deletion fails, e.g., if no model exists at `path`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function removeModel(url) {
    const schemeAndPath = parseURL(url);
    const manager = ModelStoreManagerRegistry.getManager(schemeAndPath.scheme);
    return manager.removeModel(schemeAndPath.path);
}
/**
 * Copy a model from one URL to another.
 *
 * This function supports:
 *
 * 1. Copying within a storage medium, e.g.,
 *    `tf.io.copyModel('localstorage://model-1', 'localstorage://model-2')`
 * 2. Copying between two storage mediums, e.g.,
 *    `tf.io.copyModel('localstorage://model-1', 'indexeddb://model-1')`
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Copy the model, from Local Storage to IndexedDB.
 * await tf.io.copyModel(
 *     'localstorage://demo/management/model1',
 *     'indexeddb://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Remove both models.
 * await tf.io.removeModel('localstorage://demo/management/model1');
 * await tf.io.removeModel('indexeddb://demo/management/model1');
 * ```
 *
 * @param sourceURL Source URL of copying.
 * @param destURL Destination URL of copying.
 * @returns ModelArtifactsInfo of the copied model (if and only if copying
 *   is successful).
 * @throws Error if copying fails, e.g., if no model exists at `sourceURL`, or
 *   if `oldPath` and `newPath` are identical.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function copyModel(sourceURL, destURL) {
    const deleteSource = false;
    return cloneModelInternal(sourceURL, destURL, deleteSource);
}
/**
 * Move a model from one URL to another.
 *
 * This function supports:
 *
 * 1. Moving within a storage medium, e.g.,
 *    `tf.io.moveModel('localstorage://model-1', 'localstorage://model-2')`
 * 2. Moving between two storage mediums, e.g.,
 *    `tf.io.moveModel('localstorage://model-1', 'indexeddb://model-1')`
 *
 * ```js
 * // First create and save a model.
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * await model.save('localstorage://demo/management/model1');
 *
 * // Then list existing models.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Move the model, from Local Storage to IndexedDB.
 * await tf.io.moveModel(
 *     'localstorage://demo/management/model1',
 *     'indexeddb://demo/management/model1');
 *
 * // List models again.
 * console.log(JSON.stringify(await tf.io.listModels()));
 *
 * // Remove the moved model.
 * await tf.io.removeModel('indexeddb://demo/management/model1');
 * ```
 *
 * @param sourceURL Source URL of moving.
 * @param destURL Destination URL of moving.
 * @returns ModelArtifactsInfo of the copied model (if and only if copying
 *   is successful).
 * @throws Error if moving fails, e.g., if no model exists at `sourceURL`, or
 *   if `oldPath` and `newPath` are identical.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Management',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
async function moveModel(sourceURL, destURL) {
    const deleteSource = true;
    return cloneModelInternal(sourceURL, destURL, deleteSource);
}

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
class PlatformBrowser {
    fetch(path, init) {
        return fetch(path, init);
    }
    now() {
        return performance.now();
    }
    encode(text, encoding) {
        if (encoding !== 'utf-8' && encoding !== 'utf8') {
            throw new Error(`Browser's encoder only supports utf-8, but got ${encoding}`);
        }
        if (this.textEncoder == null) {
            this.textEncoder = new TextEncoder();
        }
        return this.textEncoder.encode(text);
    }
    decode(bytes, encoding) {
        return new TextDecoder(encoding).decode(bytes);
    }
}
if (env().get('IS_BROWSER')) {
    env().setPlatform('browser', new PlatformBrowser());
    // Register LocalStorage IOHandler
    try {
        ModelStoreManagerRegistry.registerManager(BrowserLocalStorage.URL_SCHEME, new BrowserLocalStorageManager());
    }
    catch (err) {
    }
    // Register IndexedDB IOHandler
    try {
        ModelStoreManagerRegistry.registerManager(BrowserIndexedDB.URL_SCHEME, new BrowserIndexedDBManager());
    }
    catch (err) {
    }
}

// We are wrapping this within an object so it can be stubbed by Jasmine.
const getNodeFetch = {
    // tslint:disable-next-line:no-require-imports
    importFetch: () => require('node-fetch')
};
let systemFetch;
class PlatformNode {
    constructor() {
        // tslint:disable-next-line:no-require-imports
        this.util = require('util');
        // According to the spec, the built-in encoder can do only UTF-8 encoding.
        // https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/TextEncoder
        this.textEncoder = new this.util.TextEncoder();
    }
    fetch(path, requestInits) {
        if (env().global.fetch != null) {
            return env().global.fetch(path, requestInits);
        }
        if (systemFetch == null) {
            systemFetch = getNodeFetch.importFetch();
        }
        return systemFetch(path, requestInits);
    }
    now() {
        const time = process.hrtime();
        return time[0] * 1000 + time[1] / 1000000;
    }
    encode(text, encoding) {
        if (encoding !== 'utf-8' && encoding !== 'utf8') {
            throw new Error(`Node built-in encoder only supports utf-8, but got ${encoding}`);
        }
        return this.textEncoder.encode(text);
    }
    decode(bytes, encoding) {
        if (bytes.length === 0) {
            return '';
        }
        return new this.util.TextDecoder(encoding).decode(bytes);
    }
}
if (env().get('IS_NODE')) {
    env().setPlatform('node', new PlatformNode());
}

/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
 * Creates an empty `tf.TensorBuffer` with the specified `shape` and `dtype`.
 *
 * The values are stored in CPU as `TypedArray`. Fill the buffer using
 * `buffer.set()`, or by modifying directly `buffer.values`.
 *
 * When done, call `buffer.toTensor()` to get an immutable `tf.Tensor` with
 * those values.
 *
 * ```js
 * // Create a buffer and set values at particular indices.
 * const buffer = tf.buffer([2, 2]);
 * buffer.set(3, 0, 0);
 * buffer.set(5, 1, 0);
 *
 * // Convert the buffer back to a tensor.
 * buffer.toTensor().print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The dtype of the buffer. Defaults to 'float32'.
 * @param values The values of the buffer as `TypedArray`. Defaults to
 * zeros.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function buffer(shape, dtype = 'float32', values) {
    dtype = dtype || 'float32';
    assertNonNegativeIntegerDimensions(shape);
    return new TensorBuffer(shape, dtype, values);
}

/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
 * Casts a `tf.Tensor` to a new dtype.
 *
 * ```js
 * const x = tf.tensor1d([1.5, 2.5, 3]);
 * tf.cast(x, 'int32').print();
 * ```
 * @param x The input tensor to be casted.
 * @param dtype The dtype to cast the input tensor to.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function cast_(x, dtype) {
    const $x = convertToTensor(x, 'x', 'cast');
    // Sanity checks.
    if (!isValidDtype(dtype)) {
        throw new Error(`Failed to cast to unknown dtype ${dtype}`);
    }
    if (dtype === 'string' && $x.dtype !== 'string' ||
        dtype !== 'string' && $x.dtype === 'string') {
        throw new Error('Only strings can be casted to strings');
    }
    const inputs = { x: $x };
    const attrs = { dtype };
    return ENGINE.runKernel(Cast, inputs, attrs);
}
const cast = op({ cast_ });

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
 * Creates a new tensor with the same values and shape as the specified
 * tensor.
 *
 * ```js
 * const x = tf.tensor([1, 2]);
 *
 * x.clone().print();
 * ```
 *
 * @param x The tensor to clone.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function clone_(x) {
    const $x = convertToTensor(x, 'x', 'clone', 'string_or_numeric');
    const inputs = { x: $x };
    // Note this op is called tf.identity in python. Hence the kernel name used
    // here.
    return ENGINE.runKernel(Identity, inputs);
}
const clone = op({ clone_ });

/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
 * Prints information about the `tf.Tensor` including its data.
 *
 * ```js
 * const verbose = true;
 * tf.tensor2d([1, 2, 3, 4], [2, 2]).print(verbose);
 * ```
 * @param x The tensor to be printed.
 * @param verbose Whether to print verbose information about the ` Tensor`,
 * including dtype and size.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function print(x, verbose = false) {
    console.log(x.toString(verbose));
}

/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
getOrMakeEngine();
const opHandler$1 = {
    buffer,
    cast,
    clone,
    print
};
setOpHandler(opHandler$1);

/**
 * Check whether updates.shape = indices.shape[:batchDim] +
 * shape[sliceDim:]
 *
 * @param x The input tensor.
 */
function validateUpdateShape(shape, indices, updates) {
    const sliceDim = (indices.rank > 1) ? indices.shape[indices.rank - 1] : 1;
    const batchDim = (indices.rank > 1) ? indices.rank - 1 : 1;
    const shapeError = 'Must have updates.shape = indices.shape[:batchDim] + ' +
        `shape[sliceDim:], got updates.shape: ${updates.shape}` +
        `, indices.shape: ${indices.shape}, shape: ${shape}` +
        `, sliceDim: ${sliceDim}, and batchDim: ${batchDim}.`;
    if (updates.rank < batchDim) {
        throw new Error(shapeError + ` update.rank < ${batchDim}. `);
    }
    if (shape.length < sliceDim + (updates.rank - batchDim)) {
        throw new Error(shapeError +
            ` Output shape length < ${sliceDim + (updates.rank - batchDim)}`);
    }
    if (updates.rank !== batchDim + shape.length - sliceDim) {
        throw new Error(shapeError + ` update.rank != ${batchDim + shape.length - sliceDim}`);
    }
    for (let d = 0; d < batchDim; ++d) {
        if (updates.shape[d] !== indices.shape[d]) {
            throw new Error(shapeError +
                ` updates.shape[${d}] (${updates.shape[d]}) != indices.shape[${d}] (${indices.shape[d]}).`);
        }
    }
    for (let d = 0; d < updates.rank - batchDim; ++d) {
        if (updates.shape[d + batchDim] !== shape[d + sliceDim]) {
            throw new Error(shapeError +
                ` updates.shape[${d + batchDim}] (${updates.shape[d + batchDim]}) != shape[${d + batchDim}] (${shape[d + batchDim]})`);
        }
    }
}
/**
 * Validate scatter nd inputs.
 *
 * @param update The tensor contains the update values.
 * @param indices The tensor contains the indices for the update values.
 * @param shape The shape of the output tensor.
 */
function validateInput(updates, indices, shape) {
    if (indices.rank < 1) {
        throw new Error('tf.scatterND() expects the indices to be rank 1 or higher,' +
            ` but the rank was ${indices.rank}.`);
    }
    if (updates.rank < 1) {
        throw new Error('tf.scatterND() expects the updates to be rank 1 or higher,' +
            ` but the rank was ${updates.rank}.`);
    }
    if (indices.dtype !== 'int32') {
        throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${indices.dtype}`);
    }
    if (shape.length < 1) {
        throw new Error(`Output rank must be greater or equal to 1, but got shape: ${shape}`);
    }
    if (shape.length === 0) {
        if (indices.size === 0) {
            throw new Error(`Indices specified for empty output. indices shape: ${indices.shape}`);
        }
        if (updates.size === 0) {
            throw new Error(`Updates specified for empty output. updates shape: ${updates.shape}`);
        }
    }
    validateUpdateShape(shape, indices, updates);
}
/**
 * Calculate the shape information for the output.
 *
 * @param update The tensor contains the update values.
 * @param indices The tensor contains the indices for the update values.
 * @param shape The shape of the output tensor.
 *
 * @returns ScatterShapeInfo
 */
function calculateShapes(updates, indices, shape) {
    // Calculate the number of dimensions in indices
    const indicesRank = indices.shape.length;
    const sliceRank = (indicesRank > 1) ? indices.shape[indicesRank - 1] : 1;
    // Calculate the number of elements that make up each slice of our updated
    // tensor. This allows us to work with flattened tensors and copy over whole
    // slices at a time.
    const totalNd = shape.length;
    let sliceSize = 1;
    for (let i = sliceRank; i < totalNd; ++i) {
        sliceSize *= shape[i];
    }
    const safeSliceDim = (sliceRank < 1) ? 1 : sliceRank;
    const numUpdates = sizeFromShape(indices.shape) / safeSliceDim;
    const strides = [...computeStrides(shape.slice(0, sliceRank)), 1];
    const outputSize = sizeFromShape(shape);
    return { sliceRank, numUpdates, sliceSize, strides, outputSize };
}

var scatter_nd_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    validateUpdateShape: validateUpdateShape,
    validateInput: validateInput,
    calculateShapes: calculateShapes
});

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 * Enables production mode which disables correctness checks in favor of
 * performance.
 *
 * @doc {heading: 'Environment'}
 */
function enableProdMode() {
    env().set('PROD', true);
}
/**
 * Enables debug mode which will log information about all executed kernels:
 * the elapsed time of the kernel execution, as well as the rank, shape, and
 * size of the output tensor.
 *
 * Debug mode will significantly slow down your application as it will
 * download the result of every operation to the CPU. This should not be used in
 * production. Debug mode does not affect the timing information of the kernel
 * execution as we do not measure download time in the kernel execution time.
 *
 * See also: `tf.profile`, `tf.memory`.
 *
 * @doc {heading: 'Environment'}
 */
function enableDebugMode() {
    env().set('DEBUG', true);
}
/** Globally disables deprecation warnings */
function disableDeprecationWarnings() {
    env().set('DEPRECATION_WARNINGS_ENABLED', false);
    console.warn(`TensorFlow.js deprecation warnings have been disabled.`);
}
/** Warn users about deprecated functionality. */
function deprecationWarn(msg) {
    if (env().getBool('DEPRECATION_WARNINGS_ENABLED')) {
        console.warn(msg + ' You can disable deprecation warnings with ' +
            'tf.disableDeprecationWarnings().');
    }
}
/**
 * Dispose all variables kept in backend engine.
 *
 * @doc {heading: 'Environment'}
 */
function disposeVariables() {
    ENGINE.disposeVariables();
}
/**
 * It returns the global engine that keeps track of all tensors and backends.
 *
 * @doc {heading: 'Environment'}
 */
function engine() {
    return ENGINE;
}
/**
 * Returns memory info at the current time in the program. The result is an
 * object with the following properties:
 *
 * - `numBytes`: Number of bytes allocated (undisposed) at this time.
 * - `numTensors`: Number of unique tensors allocated.
 * - `numDataBuffers`: Number of unique data buffers allocated
 *   (undisposed) at this time, which is ??? the number of tensors
 *   (e.g. `a.reshape(newShape)` makes a new Tensor that shares the same
 *   data buffer with `a`).
 * - `unreliable`: True if the memory usage is unreliable. See `reasons` when
 *    `unreliable` is true.
 * - `reasons`: `string[]`, reasons why the memory is unreliable, present if
 *    `unreliable` is true.
 *
 * WebGL Properties:
 * - `numBytesInGPU`: Number of bytes allocated (undisposed) in the GPU only at
 *     this time.
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
function memory() {
    return ENGINE.memory();
}
/**
 * Executes the provided function `f()` and returns a promise that resolves
 * with information about the function's memory use:
 * - `newBytes`: the number of new bytes allocated
 * - `newTensors`: the number of new tensors created
 * - `peakBytes`: the peak number of bytes allocated
 * - `kernels`: an array of objects for each kernel involved that reports
 * their input and output shapes, number of bytes used, and number of new
 * tensors created.
 * - `kernelNames`: an array of unique strings with just the names of the
 * kernels in the `kernels` array.
 *
 * ```js
 * const profile = await tf.profile(() => {
 *   const x = tf.tensor1d([1, 2, 3]);
 *   let x2 = x.square();
 *   x2.dispose();
 *   x2 = x.square();
 *   x2.dispose();
 *   return x;
 * });
 *
 * console.log(`newBytes: ${profile.newBytes}`);
 * console.log(`newTensors: ${profile.newTensors}`);
 * console.log(`byte usage over all kernels: ${profile.kernels.map(k =>
 * k.totalBytesSnapshot)}`);
 * ```
 *
 *
 * @doc {heading: 'Performance', subheading: 'Profile'}
 */
function profile(f) {
    return ENGINE.profile(f);
}
/**
 * Executes the provided function `fn` and after it is executed, cleans up all
 * intermediate tensors allocated by `fn` except those returned by `fn`.
 * `fn` must not return a Promise (async functions not allowed). The returned
 * result can be a complex object.
 *
 * Using this method helps avoid memory leaks. In general, wrap calls to
 * operations in `tf.tidy` for automatic memory cleanup.
 *
 * NOTE: Variables do *not* get cleaned up when inside a tidy(). If you want to
 * dispose variables, please use `tf.disposeVariables` or call dispose()
 * directly on variables.
 *
 * ```js
 * // y = 2 ^ 2 + 1
 * const y = tf.tidy(() => {
 *   // a, b, and one will be cleaned up when the tidy ends.
 *   const one = tf.scalar(1);
 *   const a = tf.scalar(2);
 *   const b = a.square();
 *
 *   console.log('numTensors (in tidy): ' + tf.memory().numTensors);
 *
 *   // The value returned inside the tidy function will return
 *   // through the tidy, in this case to the variable y.
 *   return b.add(one);
 * });
 *
 * console.log('numTensors (outside tidy): ' + tf.memory().numTensors);
 * y.print();
 * ```
 *
 * @param nameOrFn The name of the closure, or the function to execute.
 *     If a name is provided, the 2nd argument should be the function.
 *     If debug mode is on, the timing and the memory usage of the function
 *     will be tracked and displayed on the console using the provided name.
 * @param fn The function to execute.
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
function tidy(nameOrFn, fn) {
    return ENGINE.tidy(nameOrFn, fn);
}
/**
 * Disposes any `tf.Tensor`s found within the provided object.
 *
 * @param container an object that may be a `tf.Tensor` or may directly
 *     contain `tf.Tensor`s, such as a `Tensor[]` or `{key: Tensor, ...}`. If
 *     the object is not a `tf.Tensor` or does not contain `Tensors`, nothing
 *     happens. In general it is safe to pass any object here, except that
 *     `Promise`s are not supported.
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
function dispose(container) {
    const tensors = getTensorsInContainer(container);
    tensors.forEach(tensor => tensor.dispose());
}
/**
 * Keeps a `tf.Tensor` generated inside a `tf.tidy` from being disposed
 * automatically.
 *
 * ```js
 * let b;
 * const y = tf.tidy(() => {
 *   const one = tf.scalar(1);
 *   const a = tf.scalar(2);
 *
 *   // b will not be cleaned up by the tidy. a and one will be cleaned up
 *   // when the tidy ends.
 *   b = tf.keep(a.square());
 *
 *   console.log('numTensors (in tidy): ' + tf.memory().numTensors);
 *
 *   // The value returned inside the tidy function will return
 *   // through the tidy, in this case to the variable y.
 *   return b.add(one);
 * });
 *
 * console.log('numTensors (outside tidy): ' + tf.memory().numTensors);
 * console.log('y:');
 * y.print();
 * console.log('b:');
 * b.print();
 * ```
 *
 * @param result The tensor to keep from being disposed.
 *
 * @doc {heading: 'Performance', subheading: 'Memory'}
 */
function keep(result) {
    return ENGINE.keep(result);
}
/**
 * Executes `f()` and returns a promise that resolves with timing
 * information.
 *
 * The result is an object with the following properties:
 *
 * - `wallMs`: Wall execution time.
 * - `kernelMs`: Kernel execution time, ignoring data transfer. If using the
 * WebGL backend and the query timer extension is not available, this will
 * return an error object.
 * - On `WebGL` The following additional properties exist:
 *   - `uploadWaitMs`: CPU blocking time on texture uploads.
 *   - `downloadWaitMs`: CPU blocking time on texture downloads (readPixels).
 *
 * ```js
 * const x = tf.randomNormal([20, 20]);
 * const time = await tf.time(() => x.matMul(x));
 *
 * console.log(`kernelMs: ${time.kernelMs}, wallTimeMs: ${time.wallMs}`);
 * ```
 *
 * @param f The function to execute and time.
 *
 * @doc {heading: 'Performance', subheading: 'Timing'}
 */
function time(f) {
    return ENGINE.time(f);
}
/**
 * Sets the backend (cpu, webgl, wasm, etc) responsible for creating tensors and
 * executing operations on those tensors. Returns a promise that resolves
 * to a boolean if the backend initialization was successful.
 *
 * Note this disposes the current backend, if any, as well as any tensors
 * associated with it. A new backend is initialized, even if it is of the
 * same type as the previous one.
 *
 * @param backendName The name of the backend. Currently supports
 *     `'webgl'|'cpu'` in the browser, `'tensorflow'` under node.js
 *     (requires tfjs-node), and `'wasm'` (requires tfjs-backend-wasm).
 *
 * @doc {heading: 'Backends'}
 */
function setBackend(backendName) {
    return ENGINE.setBackend(backendName);
}
/**
 * Returns a promise that resolves when the currently selected backend (or the
 * highest priority one) has initialized. Await this promise when you are using
 * a backend that has async initialization.
 *
 * @doc {heading: 'Backends'}
 */
function ready() {
    return ENGINE.ready();
}
/**
 * Returns the current backend name (cpu, webgl, etc). The backend is
 * responsible for creating tensors and executing operations on those tensors.
 *
 * @doc {heading: 'Backends'}
 */
function getBackend() {
    return ENGINE.backendName;
}
/**
 * Removes a backend and the registered factory.
 *
 * @doc {heading: 'Backends'}
 */
function removeBackend(name) {
    ENGINE.removeBackend(name);
}
/**
 * Finds the backend registered under the provided name. Returns null if the
 * name is not in the registry, or the registration hasn't finished yet.
 */
function findBackend(name) {
    return ENGINE.findBackend(name);
}
/**
 * Finds the backend factory registered under the provided name. Returns a
 * function that produces a new backend when called. Returns null if the name
 * is not in the registry.
 */
function findBackendFactory(name) {
    return ENGINE.findBackendFactory(name);
}
/**
 * Registers a global backend. The registration should happen when importing
 * a module file (e.g. when importing `backend_webgl.ts`), and is used for
 * modular builds (e.g. custom tfjs bundle with only webgl support).
 *
 * @param factory The backend factory function. When called, it should
 * return a backend instance, or a promise of an instance.
 * @param priority The priority of the backend (higher = more important).
 *     In case multiple backends are registered, the priority is used to find
 *     the best backend. Defaults to 1.
 * @return False if there is already a registered backend under this name, true
 *     if not.
 *
 * @doc {heading: 'Backends'}
 */
function registerBackend(name, factory, priority = 1) {
    return ENGINE.registerBackend(name, factory, priority);
}
/**
 * Gets the current backend. If no backends have been initialized, this will
 * attempt to initialize the best backend. Will throw an error if the highest
 * priority backend has async initialization, in which case, you should call
 * 'await tf.ready()' before running other code.
 *
 * @doc {heading: 'Backends'}
 */
function backend() {
    return ENGINE.backend;
}
/**
 * Sets the global platform.
 *
 * @param platformName The name of this platform.
 * @param platform A platform implementation.
 */
function setPlatform(platformName, platform) {
    env().setPlatform(platformName, platform);
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
 * Multiplies two `tf.Tensor`s element-wise, A * B. Supports broadcasting.
 *
 * We also expose `tf.mulStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3, 4]);
 * const b = tf.tensor1d([2, 3, 4, 5]);
 *
 * a.mul(b).print();  // or tf.mul(a, b)
 * ```
 *
 * ```js
 * // Broadcast mul a with b.
 * const a = tf.tensor1d([1, 2, 3, 4]);
 * const b = tf.scalar(5);
 *
 * a.mul(b).print();  // or tf.mul(a, b)
 * ```
 * @param a The first tensor to multiply.
 * @param b The second tensor to multiply. Must have the same dtype as `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function mul_(a, b) {
    let $a = convertToTensor(a, 'a', 'mul');
    let $b = convertToTensor(b, 'b', 'mul');
    [$a, $b] = makeTypesMatch($a, $b);
    const inputs = { a: $a, b: $b };
    return ENGINE.runKernel(Multiply, inputs);
}
const mul = op({ mul_ });

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
 *
 * @param inputShape Input tensor shape is of the following dimensions:
 *     `[batch, height, width, inChannels]`.
 * @param filterShape The filter shape is of the following dimensions:
 *     `[filterHeight, filterWidth, depth]`.
 * @param strides The strides of the sliding window for each dimension of the
 *     input tensor: `[strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat The data format of the input and output data.
 *     Defaults to 'NHWC'.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`.
 *     Defaults to `[1, 1]`. If `dilations` is a single number, then
 *     `dilationHeight == dilationWidth`.
 */
function computeDilation2DInfo(inputShape, filterShape, strides, pad, dataFormat = 'NHWC', dilations) {
    // `computerConv2DInfo` require filterShape to be in the dimension of:
    // `[filterHeight, filterWidth, depth, outDepth]`, dilation2d doesn't have
    // outDepth, it should have the same depth as the input.
    // Input shape: [batch, height, width, inChannels]
    const inputChannels = inputShape[3];
    const $filterShape = [...filterShape, inputChannels];
    const $dataFormat = convertConv2DDataFormat(dataFormat);
    return computeConv2DInfo(inputShape, $filterShape, strides, dilations, pad, null /* roundingMode */, null /* depthWise */, $dataFormat);
}
function computePool2DInfo(inShape, filterSize, strides, dilations, pad, roundingMode, dataFormat = 'channelsLast') {
    const [filterHeight, filterWidth] = parseTupleParam(filterSize);
    let filterShape;
    if (dataFormat === 'channelsLast') {
        filterShape = [filterHeight, filterWidth, inShape[3], inShape[3]];
    }
    else if (dataFormat === 'channelsFirst') {
        filterShape = [filterHeight, filterWidth, inShape[1], inShape[1]];
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
    return computeConv2DInfo(inShape, filterShape, strides, dilations, pad, roundingMode, false, dataFormat);
}
/**
 * Computes the information for a forward pass of a pooling3D operation.
 */
function computePool3DInfo(inShape, filterSize, strides, dilations, pad, roundingMode, dataFormat = 'NDHWC') {
    const [filterDepth, filterHeight, filterWidth] = parse3TupleParam(filterSize);
    let filterShape;
    let $dataFormat;
    if (dataFormat === 'NDHWC') {
        $dataFormat = 'channelsLast';
        filterShape =
            [filterDepth, filterHeight, filterWidth, inShape[4], inShape[4]];
    }
    else if (dataFormat === 'NCDHW') {
        $dataFormat = 'channelsFirst';
        filterShape =
            [filterDepth, filterHeight, filterWidth, inShape[1], inShape[1]];
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
    return computeConv3DInfo(inShape, filterShape, strides, dilations, pad, false, $dataFormat, roundingMode);
}
/**
 * Computes the information for a forward pass of a convolution/pooling
 * operation.
 */
function computeConv2DInfo(inShape, filterShape, strides, dilations, pad, roundingMode, depthwise = false, dataFormat = 'channelsLast') {
    let [batchSize, inHeight, inWidth, inChannels] = [-1, -1, -1, -1];
    if (dataFormat === 'channelsLast') {
        [batchSize, inHeight, inWidth, inChannels] = inShape;
    }
    else if (dataFormat === 'channelsFirst') {
        [batchSize, inChannels, inHeight, inWidth] = inShape;
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
    const [filterHeight, filterWidth, , filterChannels] = filterShape;
    const [strideHeight, strideWidth] = parseTupleParam(strides);
    const [dilationHeight, dilationWidth] = parseTupleParam(dilations);
    const effectiveFilterHeight = getEffectiveFilterSize(filterHeight, dilationHeight);
    const effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
    const { padInfo, outHeight, outWidth } = getPadAndOutInfo(pad, inHeight, inWidth, strideHeight, strideWidth, effectiveFilterHeight, effectiveFilterWidth, roundingMode, dataFormat);
    const outChannels = depthwise ? filterChannels * inChannels : filterChannels;
    let outShape;
    if (dataFormat === 'channelsFirst') {
        outShape = [batchSize, outChannels, outHeight, outWidth];
    }
    else if (dataFormat === 'channelsLast') {
        outShape = [batchSize, outHeight, outWidth, outChannels];
    }
    return {
        batchSize,
        dataFormat,
        inHeight,
        inWidth,
        inChannels,
        outHeight,
        outWidth,
        outChannels,
        padInfo,
        strideHeight,
        strideWidth,
        filterHeight,
        filterWidth,
        effectiveFilterHeight,
        effectiveFilterWidth,
        dilationHeight,
        dilationWidth,
        inShape,
        outShape,
        filterShape
    };
}
/**
 * Computes the information for a forward pass of a 3D convolution/pooling
 * operation.
 */
function computeConv3DInfo(inShape, filterShape, strides, dilations, pad, depthwise = false, dataFormat = 'channelsLast', roundingMode) {
    let [batchSize, inDepth, inHeight, inWidth, inChannels] = [-1, -1, -1, -1, -1];
    if (dataFormat === 'channelsLast') {
        [batchSize, inDepth, inHeight, inWidth, inChannels] = inShape;
    }
    else if (dataFormat === 'channelsFirst') {
        [batchSize, inChannels, inDepth, inHeight, inWidth] = inShape;
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
    }
    const [filterDepth, filterHeight, filterWidth, , filterChannels] = filterShape;
    const [strideDepth, strideHeight, strideWidth] = parse3TupleParam(strides);
    const [dilationDepth, dilationHeight, dilationWidth] = parse3TupleParam(dilations);
    const effectiveFilterDepth = getEffectiveFilterSize(filterDepth, dilationDepth);
    const effectiveFilterHeight = getEffectiveFilterSize(filterHeight, dilationHeight);
    const effectiveFilterWidth = getEffectiveFilterSize(filterWidth, dilationWidth);
    const { padInfo, outDepth, outHeight, outWidth } = get3DPadAndOutInfo(pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth, effectiveFilterDepth, effectiveFilterHeight, effectiveFilterWidth, roundingMode);
    const outChannels = depthwise ? filterChannels * inChannels : filterChannels;
    let outShape;
    if (dataFormat === 'channelsFirst') {
        outShape = [batchSize, outChannels, outDepth, outHeight, outWidth];
    }
    else if (dataFormat === 'channelsLast') {
        outShape = [batchSize, outDepth, outHeight, outWidth, outChannels];
    }
    return {
        batchSize,
        dataFormat,
        inDepth,
        inHeight,
        inWidth,
        inChannels,
        outDepth,
        outHeight,
        outWidth,
        outChannels,
        padInfo,
        strideDepth,
        strideHeight,
        strideWidth,
        filterDepth,
        filterHeight,
        filterWidth,
        effectiveFilterDepth,
        effectiveFilterHeight,
        effectiveFilterWidth,
        dilationDepth,
        dilationHeight,
        dilationWidth,
        inShape,
        outShape,
        filterShape
    };
}
function computeOutputShape2D(inShape, fieldSize, stride, zeroPad, roundingMode) {
    if (zeroPad == null) {
        zeroPad = computeDefaultPad(inShape, fieldSize, stride);
    }
    const inputRows = inShape[0];
    const inputCols = inShape[1];
    const outputRows = round((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
    const outputCols = round((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
    return [outputRows, outputCols];
}
function computeOutputShape4D(inShape, fieldSize, outChannels, stride, zeroPad, roundingMode) {
    if (zeroPad == null) {
        zeroPad = computeDefaultPad(inShape, fieldSize, stride);
    }
    const inputDepth = inShape[0];
    const inputRows = inShape[1];
    const inputCols = inShape[2];
    const outputDepths = round((inputDepth - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
    const outputRows = round((inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
    const outputCols = round((inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
    return [outputDepths, outputRows, outputCols, outChannels];
}
function computeDefaultPad(inputShape, fieldSize, stride, dilation = 1) {
    const effectiveFieldSize = getEffectiveFilterSize(fieldSize, dilation);
    return Math.floor((inputShape[0] * (stride - 1) - stride + effectiveFieldSize) / 2);
}
function parseTupleParam(param) {
    if (typeof param === 'number') {
        return [param, param, param];
    }
    if (param.length === 2) {
        return [param[0], param[1], 1];
    }
    return param;
}
function parse3TupleParam(param) {
    return typeof param === 'number' ? [param, param, param] : param;
}
/* See https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
 * Atrous convolution is equivalent to standard convolution with upsampled
 * filters with effective_filter_height =
 * filter_height + (filter_height - 1) * (dilation - 1)
 * and effective_filter_width =
 * filter_width + (filter_width - 1) * (dilation - 1),
 * produced by inserting dilation - 1 zeros along consecutive elements across
 * the filters' spatial dimensions.
 * When there is a dilation, this converts a filter dimension to the
 * effective filter dimension, so it can be used in a standard convolution.
 */
function getEffectiveFilterSize(filterSize, dilation) {
    if (dilation <= 1) {
        return filterSize;
    }
    return filterSize + (filterSize - 1) * (dilation - 1);
}
function getPadAndOutInfo(pad, inHeight, inWidth, strideHeight, strideWidth, filterHeight, filterWidth, roundingMode, dataFormat) {
    let padInfo;
    let outHeight;
    let outWidth;
    if (typeof pad === 'number') {
        const padType = (pad === 0) ? 'VALID' : 'NUMBER';
        padInfo = { top: pad, bottom: pad, left: pad, right: pad, type: padType };
        const outShape = computeOutputShape2D([inHeight, inWidth], filterHeight, strideHeight, pad, roundingMode);
        outHeight = outShape[0];
        outWidth = outShape[1];
    }
    else if (pad === 'same') {
        outHeight = Math.ceil(inHeight / strideHeight);
        outWidth = Math.ceil(inWidth / strideWidth);
        const padAlongHeight = Math.max(0, (outHeight - 1) * strideHeight + filterHeight - inHeight);
        const padAlongWidth = Math.max(0, (outWidth - 1) * strideWidth + filterWidth - inWidth);
        const top = Math.floor(padAlongHeight / 2);
        const bottom = padAlongHeight - top;
        const left = Math.floor(padAlongWidth / 2);
        const right = padAlongWidth - left;
        padInfo = { top, bottom, left, right, type: 'SAME' };
    }
    else if (pad === 'valid') {
        padInfo = { top: 0, bottom: 0, left: 0, right: 0, type: 'VALID' };
        outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
        outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
    }
    else if (typeof pad === 'object') {
        const top = dataFormat === 'channelsLast' ? pad[1][0] : pad[2][0];
        const bottom = dataFormat === 'channelsLast' ? pad[1][1] : pad[2][1];
        const left = dataFormat === 'channelsLast' ? pad[2][0] : pad[3][0];
        const right = dataFormat === 'channelsLast' ? pad[2][1] : pad[3][1];
        const padType = (top === 0 && bottom === 0 && left === 0 && right === 0) ?
            'VALID' :
            'EXPLICIT';
        padInfo = { top, bottom, left, right, type: padType };
        outHeight = round((inHeight - filterHeight + top + bottom) / strideHeight + 1, roundingMode);
        outWidth = round((inWidth - filterWidth + left + right) / strideWidth + 1, roundingMode);
    }
    else {
        throw Error(`Unknown padding parameter: ${pad}`);
    }
    return { padInfo, outHeight, outWidth };
}
function get3DPadAndOutInfo(pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth, filterDepth, filterHeight, filterWidth, roundingMode) {
    let padInfo;
    let outDepth;
    let outHeight;
    let outWidth;
    if (typeof pad === 'number') {
        const padType = (pad === 0) ? 'VALID' : 'NUMBER';
        padInfo = {
            top: pad,
            bottom: pad,
            left: pad,
            right: pad,
            front: pad,
            back: pad,
            type: padType
        };
        const outShape = computeOutputShape4D([inDepth, inHeight, inWidth, 1], filterDepth, 1, strideDepth, pad, roundingMode);
        outDepth = outShape[0];
        outHeight = outShape[1];
        outWidth = outShape[2];
    }
    else if (pad === 'same') {
        outDepth = Math.ceil(inDepth / strideDepth);
        outHeight = Math.ceil(inHeight / strideHeight);
        outWidth = Math.ceil(inWidth / strideWidth);
        const padAlongDepth = (outDepth - 1) * strideDepth + filterDepth - inDepth;
        const padAlongHeight = (outHeight - 1) * strideHeight + filterHeight - inHeight;
        const padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
        const front = Math.floor(padAlongDepth / 2);
        const back = padAlongDepth - front;
        const top = Math.floor(padAlongHeight / 2);
        const bottom = padAlongHeight - top;
        const left = Math.floor(padAlongWidth / 2);
        const right = padAlongWidth - left;
        padInfo = { top, bottom, left, right, front, back, type: 'SAME' };
    }
    else if (pad === 'valid') {
        padInfo = {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
            front: 0,
            back: 0,
            type: 'VALID'
        };
        outDepth = Math.ceil((inDepth - filterDepth + 1) / strideDepth);
        outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
        outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
    }
    else {
        throw Error(`Unknown padding parameter: ${pad}`);
    }
    return { padInfo, outDepth, outHeight, outWidth };
}
/**
 * Rounds a value depending on the rounding mode
 * @param value
 * @param roundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function round(value, roundingMode) {
    if (!roundingMode) {
        return Math.trunc(value);
    }
    switch (roundingMode) {
        case 'round':
            // used for Caffe Conv
            return Math.round(value);
        case 'ceil':
            // used for Caffe Pool
            return Math.ceil(value);
        case 'floor':
            return Math.floor(value);
        default:
            throw new Error(`Unknown roundingMode ${roundingMode}`);
    }
}
function tupleValuesAreOne(param) {
    const [dimA, dimB, dimC] = parseTupleParam(param);
    return dimA === 1 && dimB === 1 && dimC === 1;
}
function eitherStridesOrDilationsAreOne(strides, dilations) {
    return tupleValuesAreOne(strides) || tupleValuesAreOne(dilations);
}
/**
 * Convert Conv2D dataFormat from 'NHWC'|'NCHW' to
 *    'channelsLast'|'channelsFirst'
 * @param dataFormat in 'NHWC'|'NCHW' mode
 * @return dataFormat in 'channelsLast'|'channelsFirst' mode
 * @throws unknown dataFormat
 */
function convertConv2DDataFormat(dataFormat) {
    if (dataFormat === 'NHWC') {
        return 'channelsLast';
    }
    else if (dataFormat === 'NCHW') {
        return 'channelsFirst';
    }
    else {
        throw new Error(`Unknown dataFormat ${dataFormat}`);
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
/**
 * Reshapes a `tf.Tensor` to a given shape.
 *
 * Given an input tensor, returns a new tensor with the same values as the
 * input tensor with shape `shape`.
 *
 * If one component of shape is the special value -1, the size of that
 * dimension is computed so that the total size remains constant. In
 * particular, a shape of [-1] flattens into 1-D. At most one component of
 * shape can be -1.
 *
 * If shape is 1-D or higher, then the operation returns a tensor with shape
 * shape filled with the values of tensor. In this case, the number of
 * elements implied by shape must be the same as the number of elements in
 * tensor.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * x.reshape([2, 2]).print();
 * ```
 *
 * @param x The input tensor to be reshaped.
 * @param shape An array of integers defining the output tensor shape.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function reshape_(x, shape) {
    const $x = convertToTensor(x, 'x', 'reshape', 'string_or_numeric');
    const inputs = { x: $x };
    const attrs = { shape };
    return ENGINE.runKernel(Reshape, inputs, attrs);
}
const reshape = op({ reshape_ });

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
/**
 * Returns the dimensions in the input shape that are broadcasted to
 * produce the provided output shape.
 *
 * The returned dimensions are 0-indexed and sorted. An example:
 * inShape = [4, 1, 3]
 * outShape = [5, 4, 3, 3]
 * result = [1]. Dimension 1 (2nd dimension of input) gets broadcasted 1 => 3.
 */
function getBroadcastDims(inShape, outShape) {
    const inRank = inShape.length;
    const dims = [];
    for (let i = 0; i < inRank; i++) {
        const dim = inRank - 1 - i;
        const a = inShape[dim] || 1;
        const b = outShape[outShape.length - 1 - i] || 1;
        if (b > 1 && a === 1) {
            dims.unshift(dim);
        }
    }
    return dims;
}
/**
 * Returns the axes in the output space that should be reduced to produce
 * the input space.
 */
function getReductionAxes(inShape, outShape) {
    const result = [];
    for (let i = 0; i < outShape.length; i++) {
        const inDim = inShape[inShape.length - i - 1];
        const outAxis = outShape.length - i - 1;
        const outDim = outShape[outAxis];
        if (inDim == null || (inDim === 1 && outDim > 1)) {
            result.unshift(outAxis);
        }
    }
    return result;
}
function assertAndGetBroadcastShape(shapeA, shapeB) {
    const result = [];
    const l = Math.max(shapeA.length, shapeB.length);
    for (let i = 0; i < l; i++) {
        let a = shapeA[shapeA.length - i - 1];
        if (a == null) {
            a = 1;
        }
        let b = shapeB[shapeB.length - i - 1];
        if (b == null) {
            b = 1;
        }
        if (a === 1) {
            result.unshift(b);
        }
        else if (b === 1) {
            result.unshift(a);
        }
        else if (a !== b) {
            const errMsg = `Operands could not be broadcast together with shapes ` +
                `${shapeA} and ${shapeB}.`;
            throw Error(errMsg);
        }
        else {
            result.unshift(a);
        }
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
/**
 * Computes exponential linear element-wise: `x > 0 ? e ^ x - 1 : 0`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 1, -3, 2]);
 *
 * x.elu().print();  // or tf.elu(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function elu_(x) {
    const $x = convertToTensor(x, 'x', 'elu');
    const inputs = { x: $x };
    return ENGINE.runKernel(Elu, inputs);
}
const elu = op({ elu_ });

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
 * Computes leaky rectified linear element-wise.
 *
 * See
 * [http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf](
 *     http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.leakyRelu(0.1).print();  // or tf.leakyRelu(x, 0.1)
 * ```
 * @param x The input tensor.
 * @param alpha The scaling factor for negative values, defaults to 0.2.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function leakyRelu_(x, alpha = 0.2) {
    const $x = convertToTensor(x, 'x', 'leakyRelu');
    const inputs = { x: $x };
    const attrs = { alpha };
    return ENGINE.runKernel(LeakyRelu, inputs, attrs);
}
const leakyRelu = op({ leakyRelu_ });

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 * Computes the sum of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces the input along the dimensions given in `axes`. Unless `keepDims`
 * is true, the rank of the `tf.Tensor` is reduced by 1 for each entry in
 * `axes`. If `keepDims` is true, the reduced dimensions are retained with
 * length 1. If axes has no entries, all dimensions are reduced, and a
 * `tf.Tensor` with a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.sum().print();  // or tf.sum(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.sum(axis).print();  // or tf.sum(x, axis)
 * ```
 *
 * @param x The input tensor to compute the sum over. If the dtype is `bool`
 *   it will be converted to `int32` and the output dtype will be `int32`.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function sum_(x, axis = null, keepDims = false) {
    let $x = convertToTensor(x, 'x', 'sum');
    if ($x.dtype === 'bool') {
        $x = cast($x, 'int32');
    }
    const inputs = { x: $x };
    const attrs = { axis, keepDims };
    return ENGINE.runKernel(Sum, inputs, attrs);
}
const sum$1 = op({ sum_ });

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
/**
 * Returns true if the axis specifies the inner most dimensions of the
 * array.
 */
function axesAreInnerMostDims(axes, rank) {
    for (let i = 0; i < axes.length; ++i) {
        if (axes[axes.length - i - 1] !== rank - 1 - i) {
            return false;
        }
    }
    return true;
}
function combineLocations(outputLoc, reduceLoc, axes) {
    const rank = outputLoc.length + reduceLoc.length;
    const loc = [];
    let outIdx = 0;
    let reduceIdx = 0;
    for (let dim = 0; dim < rank; dim++) {
        if (axes.indexOf(dim) === -1) {
            loc.push(outputLoc[outIdx++]);
        }
        else {
            loc.push(reduceLoc[reduceIdx++]);
        }
    }
    return loc;
}
function computeOutAndReduceShapes(aShape, axes) {
    const outShape = [];
    const rank = aShape.length;
    for (let dim = 0; dim < rank; dim++) {
        if (axes.indexOf(dim) === -1) {
            outShape.push(aShape[dim]);
        }
    }
    const reduceShape = axes.map(dim => aShape[dim]);
    return [outShape, reduceShape];
}
function expandShapeToKeepDim(shape, axes) {
    const reduceSubShape = axes.map(x => 1);
    return combineLocations(shape, reduceSubShape, axes);
}
function assertAxesAreInnerMostDims(msg, axes, rank) {
    assert(axesAreInnerMostDims(axes, rank), () => `${msg} supports only inner-most axes for now. ` +
        `Got axes ${axes} and rank-${rank} input.`);
}
/**
 * Returns the axes permutation to be used with `tf.transpose`, if such
 * permutation is necessary. Otherwise it returns null. This method is used by
 * operations that operate only on inner-most axes.
 */
function getAxesPermutation(axes, rank) {
    if (axesAreInnerMostDims(axes, rank)) {
        return null;
    }
    const result = [];
    for (let i = 0; i < rank; ++i) {
        if (axes.indexOf(i) === -1) {
            result.push(i);
        }
    }
    axes.forEach(axis => result.push(axis));
    return result;
}
/** Returns the axes permutation that undoes the original permutation. */
function getUndoAxesPermutation(axes) {
    return axes.map((axis, i) => [i, axis])
        .sort((a, b) => a[1] - b[1])
        .map(x => x[0]);
}
function getInnerMostAxes(numAxes, rank) {
    const res = [];
    for (let i = rank - numAxes; i < rank; ++i) {
        res.push(i);
    }
    return res;
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
 * Computes leaky rectified linear element-wise with parametric alphas.
 *
 * `x < 0 ? alpha * x : f(x) = x`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 * const alpha = tf.scalar(0.1);
 *
 * x.prelu(alpha).print();  // or tf.prelu(x, alpha)
 * ```
 * @param x The input tensor.
 * @param alpha Scaling factor for negative values.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function prelu_(x, alpha) {
    const $x = convertToTensor(x, 'x', 'prelu');
    const $alpha = convertToTensor(alpha, 'alpha', 'prelu');
    const inputs = { x: $x, alpha: $alpha };
    return ENGINE.runKernel(Prelu, inputs);
}
const prelu = op({ prelu_ });

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
 * Computes rectified linear element-wise: `max(x, 0)`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.relu().print();  // or tf.relu(x)
 * ```
 * @param x The input tensor. If the dtype is `bool`, the output dtype will be
 *     `int32'.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function relu_(x) {
    const $x = convertToTensor(x, 'x', 'relu');
    const inputs = { x: $x };
    return ENGINE.runKernel(Relu, inputs);
}
const relu = op({ relu_ });

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
 * Computes rectified linear 6 element-wise: `min(max(x, 0), 6)`.
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 8]);
 *
 * x.relu6().print();  // or tf.relu6(x)
 * ```
 * @param x The input tensor. If the dtype is `bool`, the output dtype will be
 *     `int32'.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function relu6_(x) {
    const $x = convertToTensor(x, 'x', 'relu6');
    const inputs = { x: $x };
    return ENGINE.runKernel(Relu6, inputs);
}
const relu6 = op({ relu6_ });

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 * Computes step of the input `tf.Tensor` element-wise: `x > 0 ? 1 : alpha * x`
 *
 * ```js
 * const x = tf.tensor1d([0, 2, -1, -3]);
 *
 * x.step(.5).print();  // or tf.step(x, .5)
 * ```
 * @param x The input tensor.
 * @param alpha The gradient when input is negative.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function step_(x, alpha = 0.0) {
    const $x = convertToTensor(x, 'x', 'step');
    const inputs = { x: $x };
    const attrs = { alpha };
    return ENGINE.runKernel(Step, inputs, attrs);
}
const step = op({ step_ });

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
function whereImpl(condShape, condVals) {
    const indices = [];
    for (let i = 0; i < condVals.length; i++) {
        if (condVals[i]) {
            indices.push(i);
        }
    }
    const inBuffer = buffer(condShape, 'int32');
    const out = buffer([indices.length, condShape.length], 'int32');
    for (let i = 0; i < indices.length; i++) {
        const loc = inBuffer.indexToLoc(indices[i]);
        const offset = i * condShape.length;
        out.values.set(loc, offset);
    }
    return out.toTensor();
}

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
// Returns gradient for fused activation.
function getFusedDyActivation(dy, y, activation) {
    if (activation == null || activation === 'linear') {
        return dy;
    }
    if (activation === 'relu') {
        return mul(dy, step(y));
    }
    throw new Error(`Cannot compute gradient for fused activation ${activation}.`);
}
// Returns gradient for fused bias.
function getFusedBiasGradient(bias, dyActivation) {
    let res = dyActivation;
    const reduceAxes = getReductionAxes(bias.shape, dyActivation.shape);
    if (reduceAxes.length > 0) {
        res = sum$1(res, reduceAxes);
    }
    return reshape(res, bias.shape);
}
function applyActivation(x, activation, preluActivationWeights, leakyreluAlpha) {
    if (activation === 'linear') {
        return x;
    }
    else if (activation === 'relu') {
        return relu(x);
    }
    else if (activation === 'elu') {
        return elu(x);
    }
    else if (activation === 'relu6') {
        return relu6(x);
    }
    else if (activation === 'prelu') {
        return prelu(x, preluActivationWeights);
    }
    else if (activation === 'leakyrelu') {
        return leakyRelu(x, leakyreluAlpha);
    }
    throw new Error(`Unknown fused activation ${activation}.`);
}
// Whether we should call fused ops.
const shouldFuse = (gradientDepth, activation) => {
    const gradientMode = gradientDepth > 0;
    return !gradientMode || activation === 'linear';
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
 * Inserts a value into a sorted array. This method allows duplicate, meaning it
 * allows inserting duplicate value, in which case, the element will be inserted
 * at the lowest index of the value.
 * @param arr The array to modify.
 * @param element The element to insert.
 * @param comparator Optional. If no comparator is specified, elements are
 * compared using array_util.defaultComparator, which is suitable for Strings
 * and Numbers in ascending arrays. If the array contains multiple instances of
 * the target value, the left-most instance will be returned. To provide a
 * comparator, it should take 2 arguments to compare and return a negative,
 * zero, or a positive number.
 */
function binaryInsert(arr, element, comparator) {
    const index = binarySearch(arr, element, comparator);
    const insertionPoint = index < 0 ? -(index + 1) : index;
    arr.splice(insertionPoint, 0, element);
}
/**
 * Searches the array for the target using binary search, returns the index
 * of the found element, or position to insert if element not found. If no
 * comparator is specified, elements are compared using array_
 * util.defaultComparator, which is suitable for Strings and Numbers in
 * ascending arrays. If the array contains multiple instances of the target
 * value, the left-most instance will be returned.
 * @param arr The array to be searched in.
 * @param target The target to be searched for.
 * @param comparator Should take 2 arguments to compare and return a negative,
 *    zero, or a positive number.
 * @return Lowest index of the target value if found, otherwise the insertion
 *    point where the target should be inserted, in the form of
 *    (-insertionPoint - 1).
 */
function binarySearch(arr, target, comparator) {
    return binarySearch_(arr, target, comparator || defaultComparator);
}
/**
 * Compares its two arguments for order.
 * @param a The first element to be compared.
 * @param b The second element to be compared.
 * @return A negative number, zero, or a positive number as the first
 *     argument is less than, equal to, or greater than the second.
 */
function defaultComparator(a, b) {
    return a > b ? 1 : a < b ? -1 : 0;
}
function binarySearch_(arr, target, comparator) {
    let left = 0;
    let right = arr.length;
    let middle = 0;
    let found = false;
    while (left < right) {
        middle = left + ((right - left) >>> 1);
        const compareResult = comparator(target, arr[middle]);
        if (compareResult > 0) {
            left = middle + 1;
        }
        else {
            right = middle;
            // If compareResult is 0, the value is found. We record it is found,
            // and then keep looking because there may be duplicate.
            found = !compareResult;
        }
    }
    return found ? left : -left - 1;
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
function nonMaxSuppressionV3Impl(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold) {
    return nonMaxSuppressionImpl_(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, 0 /* softNmsSigma */);
}
function nonMaxSuppressionV4Impl(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, padToMaxOutputSize) {
    return nonMaxSuppressionImpl_(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, 0 /* softNmsSigma */, false /* returnScoresTensor */, padToMaxOutputSize /* padToMaxOutputSize */, true
    /* returnValidOutputs */ );
}
function nonMaxSuppressionV5Impl(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma) {
    return nonMaxSuppressionImpl_(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma, true /* returnScoresTensor */);
}
function nonMaxSuppressionImpl_(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold, softNmsSigma, returnScoresTensor = false, padToMaxOutputSize = false, returnValidOutputs = false) {
    // The list is sorted in ascending order, so that we can always pop the
    // candidate with the largest score in O(1) time.
    const candidates = [];
    for (let i = 0; i < scores.length; i++) {
        if (scores[i] > scoreThreshold) {
            candidates.push({ score: scores[i], boxIndex: i, suppressBeginIndex: 0 });
        }
    }
    candidates.sort(ascendingComparator);
    // If softNmsSigma is 0, the outcome of this algorithm is exactly same as
    // before.
    const scale = softNmsSigma > 0 ? (-0.5 / softNmsSigma) : 0.0;
    const selectedIndices = [];
    const selectedScores = [];
    while (selectedIndices.length < maxOutputSize && candidates.length > 0) {
        const candidate = candidates.pop();
        const { score: originalScore, boxIndex, suppressBeginIndex } = candidate;
        if (originalScore < scoreThreshold) {
            break;
        }
        // Overlapping boxes are likely to have similar scores, therefore we
        // iterate through the previously selected boxes backwards in order to
        // see if candidate's score should be suppressed. We use
        // suppressBeginIndex to track and ensure a candidate can be suppressed
        // by a selected box no more than once. Also, if the overlap exceeds
        // iouThreshold, we simply ignore the candidate.
        let ignoreCandidate = false;
        for (let j = selectedIndices.length - 1; j >= suppressBeginIndex; --j) {
            const iou = intersectionOverUnion(boxes, boxIndex, selectedIndices[j]);
            if (iou >= iouThreshold) {
                ignoreCandidate = true;
                break;
            }
            candidate.score =
                candidate.score * suppressWeight(iouThreshold, scale, iou);
            if (candidate.score <= scoreThreshold) {
                break;
            }
        }
        // At this point, if `candidate.score` has not dropped below
        // `scoreThreshold`, then we know that we went through all of the
        // previous selections and can safely update `suppressBeginIndex` to the
        // end of the selected array. Then we can re-insert the candidate with
        // the updated score and suppressBeginIndex back in the candidate list.
        // If on the other hand, `candidate.score` has dropped below the score
        // threshold, we will not add it back to the candidates list.
        candidate.suppressBeginIndex = selectedIndices.length;
        if (!ignoreCandidate) {
            // Candidate has passed all the tests, and is not suppressed, so
            // select the candidate.
            if (candidate.score === originalScore) {
                selectedIndices.push(boxIndex);
                selectedScores.push(candidate.score);
            }
            else if (candidate.score > scoreThreshold) {
                // Candidate's score is suppressed but is still high enough to be
                // considered, so add back to the candidates list.
                binaryInsert(candidates, candidate, ascendingComparator);
            }
        }
    }
    // NonMaxSuppressionV4 feature: padding output to maxOutputSize.
    const validOutputs = selectedIndices.length;
    const elemsToPad = maxOutputSize - validOutputs;
    if (padToMaxOutputSize && elemsToPad > 0) {
        selectedIndices.push(...new Array(elemsToPad).fill(0));
        selectedScores.push(...new Array(elemsToPad).fill(0.0));
    }
    const result = { selectedIndices };
    if (returnScoresTensor) {
        result['selectedScores'] = selectedScores;
    }
    if (returnValidOutputs) {
        result['validOutputs'] = validOutputs;
    }
    return result;
}
function intersectionOverUnion(boxes, i, j) {
    const iCoord = boxes.subarray(i * 4, i * 4 + 4);
    const jCoord = boxes.subarray(j * 4, j * 4 + 4);
    const yminI = Math.min(iCoord[0], iCoord[2]);
    const xminI = Math.min(iCoord[1], iCoord[3]);
    const ymaxI = Math.max(iCoord[0], iCoord[2]);
    const xmaxI = Math.max(iCoord[1], iCoord[3]);
    const yminJ = Math.min(jCoord[0], jCoord[2]);
    const xminJ = Math.min(jCoord[1], jCoord[3]);
    const ymaxJ = Math.max(jCoord[0], jCoord[2]);
    const xmaxJ = Math.max(jCoord[1], jCoord[3]);
    const areaI = (ymaxI - yminI) * (xmaxI - xminI);
    const areaJ = (ymaxJ - yminJ) * (xmaxJ - xminJ);
    if (areaI <= 0 || areaJ <= 0) {
        return 0.0;
    }
    const intersectionYmin = Math.max(yminI, yminJ);
    const intersectionXmin = Math.max(xminI, xminJ);
    const intersectionYmax = Math.min(ymaxI, ymaxJ);
    const intersectionXmax = Math.min(xmaxI, xmaxJ);
    const intersectionArea = Math.max(intersectionYmax - intersectionYmin, 0.0) *
        Math.max(intersectionXmax - intersectionXmin, 0.0);
    return intersectionArea / (areaI + areaJ - intersectionArea);
}
// A Gaussian penalty function, this method always returns values in [0, 1].
// The weight is a function of similarity, the more overlap two boxes are, the
// smaller the weight is, meaning highly overlapping boxe will be significantly
// penalized. On the other hand, a non-overlapping box will not be penalized.
function suppressWeight(iouThreshold, scale, iou) {
    const weight = Math.exp(scale * iou * iou);
    return iou <= iouThreshold ? weight : 0.0;
}
function ascendingComparator(c1, c2) {
    // For objects with same scores, we make the object with the larger index go
    // first. In an array that pops from the end, this means that the object with
    // the smaller index will be popped first. This ensures the same output as
    // the TensorFlow python version.
    return (c1.score - c2.score) ||
        ((c1.score === c2.score) && (c2.boxIndex - c1.boxIndex));
}

export { Atanh as $, concatenateArrayBuffers as A, op as B, convertToTensor as C, DTYPE_VALUE_SIZE_MAP as D, makeTypesMatch as E, ENGINE as F, BatchMatMul as G, Transpose as H, IORouterRegistry as I, Add as J, FloorDiv as K, ComplexAbs as L, Abs as M, Acos as N, OneHot as O, Acosh as P, AddN as Q, RealDiv as R, All as S, Tensor as T, Any as U, ArgMax as V, ArgMin as W, Asin as X, Asinh as Y, Atan as Z, Atan2 as _, relu as a, Min as a$, eitherStridesOrDilationsAreOne as a0, isInt as a1, AvgPool as a2, AvgPool3D as a3, convertToTensorArray as a4, Concat as a5, Sigmoid as a6, Slice as a7, Tanh as a8, BatchToSpaceND as a9, Floor as aA, GatherV2 as aB, Greater as aC, GreaterEqual as aD, Imag as aE, Less as aF, LessEqual as aG, LinSpace as aH, LRN as aI, Log as aJ, Log1p as aK, isFunction as aL, Variable as aM, Neg as aN, Softplus as aO, Max as aP, Sub as aQ, parseAxisParam as aR, expandShapeToKeepDim as aS, LogicalAnd as aT, LogicalNot as aU, LogicalOr as aV, MaxPool as aW, MaxPool3D as aX, MaxPoolWithArgmax as aY, Maximum as aZ, Mean as a_, FusedBatchNorm as aa, Bincount as ab, Tile as ac, Ceil as ad, ClipByValue as ae, Conv2D as af, Conv2DBackpropInput as ag, Conv3D as ah, Cos as ai, Cosh as aj, Cumsum as ak, DenseBincount as al, DepthToSpace as am, DepthwiseConv2dNative as an, Dilation2D as ao, assertAndGetBroadcastShape as ap, Equal as aq, assertShapesMatch as ar, Select as as, ZerosLike as at, Erf as au, Exp as av, ExpandDims as aw, Expm1 as ax, buffer as ay, Fill as az, elu as b, nonMaxSuppressionV4Impl as b$, Minimum as b0, MirrorPad as b1, Mod as b2, Multinomial as b3, NotEqual as b4, makeOnesTypedArray as b5, OnesLike as b6, PadV2 as b7, SpaceToBatchND as b8, Pow as b9, Unique as bA, Unpack as bB, whereImpl as bC, validateInput as bD, ScatterNd as bE, SparseToDense as bF, GatherNd as bG, Conv2DBackpropFilter as bH, shouldFuse as bI, applyActivation as bJ, computeConv2DInfo as bK, FusedConv2D as bL, getFusedDyActivation as bM, tupleValuesAreOne as bN, getFusedBiasGradient as bO, DepthwiseConv2dNativeBackpropFilter as bP, DepthwiseConv2dNativeBackpropInput as bQ, FusedDepthwiseConv2D as bR, _FusedMatMul as bS, CropAndResize as bT, FlipLeftRight as bU, RotateWithOffset as bV, NonMaxSuppressionV3 as bW, nonMaxSuppressionV3Impl as bX, NonMaxSuppressionV5 as bY, nonMaxSuppressionV5Impl as bZ, NonMaxSuppressionV4 as b_, Prod as ba, Range as bb, Real as bc, Reciprocal as bd, Reverse as be, Round as bf, Rsqrt as bg, Selu as bh, TensorBuffer as bi, Sign as bj, Sin as bk, Sinh as bl, Softmax as bm, FFT as bn, IFFT as bo, SplitV as bp, Sqrt as bq, SquaredDifference as br, squeezeShape as bs, Pack as bt, StridedSlice as bu, Tan as bv, assertNonNull as bw, inferShape as bx, makeTensor as by, TopK as bz, clone as c, Complex as c$, ResizeBilinear as c0, ResizeNearestNeighbor as c1, dispose as c2, makeZerosTypedArray as c3, isTypedArray as c4, KernelBackend as c5, DataStorage as c6, engine as c7, isString as c8, encodeString as c9, Diag as cA, computeDilation2DInfo as cB, getArrayFromDType as cC, locToIndex as cD, toTypedArray as cE, Dilation2DBackpropFilter as cF, makeZerosNestedTypedArray as cG, Dilation2DBackpropInput as cH, EluGrad as cI, createScalarValue as cJ, inferDtype as cK, IsFinite as cL, IsInf as cM, IsNan as cN, LRNGrad as cO, MaxPool3DGrad as cP, MaxPoolGrad as cQ, Sum as cR, indexToLoc as cS, ResizeBilinearGrad as cT, ResizeNearestNeighborGrad as cU, calculateShapes as cV, Square as cW, Step as cX, UnsortedSegmentSum as cY, registerKernel as cZ, getBroadcastDims as c_, decodeString as ca, now as cb, registerBackend as cc, Elu as cd, LeakyRelu as ce, getTypedArrayFromDType as cf, Prelu as cg, Relu as ch, Relu6 as ci, Reshape as cj, inferFromImplicitShape as ck, computeStrides as cl, getAxesPermutation as cm, getInnerMostAxes as cn, assertAxesAreInnerMostDims as co, computeOutAndReduceShapes as cp, computePool2DInfo as cq, computePool3DInfo as cr, AvgPool3DGrad as cs, AvgPoolGrad as ct, convertConv2DDataFormat as cu, computeConv3DInfo as cv, Conv3DBackpropFilterV2 as cw, Conv3DBackpropInputV2 as cx, upcastType as cy, getUndoAxesPermutation as cz, complex as d, copyRegisteredKernels as d$, Identity as d0, Cast as d1, hasEncodingLoss as d2, Multiply as d3, clamp as d4, nearestDivisor as d5, axesAreInnerMostDims as d6, combineLocations as d7, getReductionAxes as d8, computeDefaultPad as d9, Rank as dA, Environment as dB, ENV as dC, print as dD, step as dE, OP_SCOPE_SUFFIX as dF, enableProdMode as dG, enableDebugMode as dH, disableDeprecationWarnings as dI, deprecationWarn as dJ, disposeVariables as dK, memory as dL, profile as dM, time as dN, setBackend as dO, ready as dP, getBackend as dQ, removeBackend as dR, findBackend as dS, findBackendFactory as dT, backend as dU, setPlatform as dV, getGradient as dW, getKernelsForBackend as dX, registerGradient as dY, unregisterKernel as dZ, unregisterGradient as d_, validateUpdateShape as da, sizeToSquarishShape as db, rightPad as dc, nearestLargerEven as dd, isMobile as de, repeatedTry as df, sum as dg, bytesPerElement as dh, isBrowser as di, sumOutType as dj, FromPixels as dk, basename as dl, encodeWeights as dm, registerLoadRouter as dn, registerSaveRouter as dp, copyModel as dq, listModels as dr, moveModel as ds, removeModel as dt, getKernel as du, assertTypesMatch as dv, util as dw, tensor_util as dx, scatter_nd_util as dy, device_util as dz, env as e, BroadcastTo as e0, LogSoftmax as e1, Pool as e2, assert as f, tensor as g, tidy as h, reshape as i, cast as j, keep as k, leakyRelu as l, mul as m, arraysEqual as n, isPromise as o, prelu as p, getLoadHandlers as q, relu6 as r, sum$1 as s, toNestedArray as t, decodeWeights as u, getSaveHandlers as v, fetch$1 as w, flatten as x, sizeFromShape as y, getModelArtifactsInfoForJSON as z };
