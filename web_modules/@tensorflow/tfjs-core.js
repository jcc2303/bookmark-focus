import '../common/__node-resolve:empty-d9702ff5.js';
import '../common/process-2545f00a.js';
import { I as IORouterRegistry, A as concatenateArrayBuffers, dl as basename, e as env, z as getModelArtifactsInfoForJSON, u as decodeWeights, dm as encodeWeights, q as getLoadHandlers, v as getSaveHandlers, dn as registerLoadRouter, dp as registerSaveRouter, dq as copyModel, dr as listModels, ds as moveModel, dt as removeModel, B as op, C as convertToTensor, f as assert, j as cast, bw as assertNonNull, bx as inferShape, by as makeTensor, T as Tensor, du as getKernel, F as ENGINE, dk as FromPixels, n as arraysEqual, c4 as isTypedArray, x as flatten, c8 as isString, c9 as encodeString, m as mul, i as reshape, cx as Conv3DBackpropInputV2, cA as Diag, cL as IsFinite, cM as IsInf, cN as IsNan, ap as assertAndGetBroadcastShape, aR as parseAxisParam, aS as expandShapeToKeepDim, a4 as convertToTensorArray, a0 as eitherStridesOrDilationsAreOne, cq as computePool2DInfo, y as sizeFromShape, ay as buffer, a1 as isInt, cY as UnsortedSegmentSum, ar as assertShapesMatch, dv as assertTypesMatch, cf as getTypedArrayFromDType, g as tensor, c2 as dispose, h as tidy, k as keep, bX as nonMaxSuppressionV3Impl, b$ as nonMaxSuppressionV4Impl, bZ as nonMaxSuppressionV5Impl, bC as whereImpl } from '../common/non_max_suppression_impl-886052e2.js';
export { M as Abs, N as Acos, P as Acosh, J as Add, Q as AddN, S as All, U as Any, V as ArgMax, W as ArgMin, X as Asin, Y as Asinh, Z as Atan, _ as Atan2, $ as Atanh, a2 as AvgPool, a3 as AvgPool3D, cs as AvgPool3DGrad, ct as AvgPoolGrad, G as BatchMatMul, a9 as BatchToSpaceND, ab as Bincount, e0 as BroadcastTo, d1 as Cast, ad as Ceil, ae as ClipByValue, c$ as Complex, L as ComplexAbs, a5 as Concat, af as Conv2D, bH as Conv2DBackpropFilter, ag as Conv2DBackpropInput, ah as Conv3D, cw as Conv3DBackpropFilterV2, cx as Conv3DBackpropInputV2, ai as Cos, aj as Cosh, bT as CropAndResize, ak as Cumsum, c6 as DataStorage, al as DenseBincount, am as DepthToSpace, an as DepthwiseConv2dNative, bP as DepthwiseConv2dNativeBackpropFilter, bQ as DepthwiseConv2dNativeBackpropInput, cA as Diag, ao as Dilation2D, cF as Dilation2DBackpropFilter, cH as Dilation2DBackpropInput, dC as ENV, cd as Elu, cI as EluGrad, dB as Environment, aq as Equal, au as Erf, av as Exp, aw as ExpandDims, ax as Expm1, bn as FFT, az as Fill, bU as FlipLeftRight, aA as Floor, K as FloorDiv, dk as FromPixels, aa as FusedBatchNorm, bL as FusedConv2D, bR as FusedDepthwiseConv2D, bG as GatherNd, aB as GatherV2, aC as Greater, aD as GreaterEqual, bo as IFFT, d0 as Identity, aE as Imag, cL as IsFinite, cM as IsInf, cN as IsNan, c5 as KernelBackend, aI as LRN, cO as LRNGrad, ce as LeakyRelu, aF as Less, aG as LessEqual, aH as LinSpace, aJ as Log, aK as Log1p, e1 as LogSoftmax, aT as LogicalAnd, aU as LogicalNot, aV as LogicalOr, aP as Max, aW as MaxPool, aX as MaxPool3D, cP as MaxPool3DGrad, cQ as MaxPoolGrad, aY as MaxPoolWithArgmax, aZ as Maximum, a_ as Mean, a$ as Min, b0 as Minimum, b1 as MirrorPad, b2 as Mod, b3 as Multinomial, d3 as Multiply, aN as Neg, bW as NonMaxSuppressionV3, b_ as NonMaxSuppressionV4, bY as NonMaxSuppressionV5, b4 as NotEqual, dF as OP_SCOPE_SUFFIX, O as OneHot, b6 as OnesLike, bt as Pack, b7 as PadV2, e2 as Pool, b9 as Pow, cg as Prelu, ba as Prod, bb as Range, dA as Rank, bc as Real, R as RealDiv, bd as Reciprocal, ch as Relu, ci as Relu6, cj as Reshape, c0 as ResizeBilinear, cT as ResizeBilinearGrad, c1 as ResizeNearestNeighbor, cU as ResizeNearestNeighborGrad, be as Reverse, bV as RotateWithOffset, bf as Round, bg as Rsqrt, bE as ScatterNd, as as Select, bh as Selu, a6 as Sigmoid, bj as Sign, bk as Sin, bl as Sinh, a7 as Slice, bm as Softmax, aO as Softplus, b8 as SpaceToBatchND, bF as SparseToDense, bp as SplitV, bq as Sqrt, cW as Square, br as SquaredDifference, cX as Step, bu as StridedSlice, aQ as Sub, cR as Sum, bv as Tan, a8 as Tanh, T as Tensor, bi as TensorBuffer, ac as Tile, bz as TopK, H as Transpose, bA as Unique, bB as Unpack, cY as UnsortedSegmentSum, aM as Variable, at as ZerosLike, bS as _FusedMatMul, dU as backend, ay as buffer, j as cast, c as clone, d as complex, d$ as copyRegisteredKernels, dJ as deprecationWarn, dz as device_util, dI as disableDeprecationWarnings, c2 as dispose, dK as disposeVariables, b as elu, dH as enableDebugMode, dG as enableProdMode, c7 as engine, e as env, dS as findBackend, dT as findBackendFactory, dQ as getBackend, dW as getGradient, du as getKernel, dX as getKernelsForBackend, k as keep, l as leakyRelu, dL as memory, m as mul, B as op, p as prelu, dD as print, dM as profile, dP as ready, cc as registerBackend, dY as registerGradient, cZ as registerKernel, a as relu, r as relu6, dR as removeBackend, i as reshape, dy as scatter_util, dO as setBackend, dV as setPlatform, dE as step, s as sum, dj as sumOutType, g as tensor, dx as tensor_util, h as tidy, dN as time, d_ as unregisterGradient, dZ as unregisterKernel, cy as upcastType, dw as util } from '../common/non_max_suppression_impl-886052e2.js';
import { b8 as browserHTTPRequest, ba as http, bb as isHTTPScheme, bc as loadWeights, bd as weightsLoaderFactory, af as oneHot, aA as transpose, aB as matMul, V as concat, h as add, X as slice, w as sigmoid, k as tanh, aG as batchNorm, be as customGrad, B as neg, o as softplus, as as logicalAnd, aq as logicalOr, ar as logicalNot, aP as mean, l as square, b as sub, b5 as pad, b4 as spaceToBatchND, a1 as avgPool, a0 as maxPool, b3 as batchToSpaceND, bf as RandGamma, bg as MPRandGauss, aX as reverse, a3 as depthwiseConv2d, a7 as conv2d, ak as whereAsync, aV as squeeze, aY as gather, c as div, p as pow, F as floor, ac as randomUniform, a6 as conv2d$1, a5 as depthwiseConv2d$1, az as matMul$1, bh as variableGrads, a9 as zerosLike, n as sqrt, ai as fill, T as abs, m as maximum } from '../common/ops-5fc79cd7.js';
export { bi as Reduction, T as abs, S as acos, R as acosh, h as add, g as addN, aN as all, aM as any, aL as argMax, aK as argMin, Q as asin, P as asinh, O as atan, N as atan2, M as atanh, a1 as avgPool, _ as avgPool3d, aG as batchNorm, b3 as batchToSpaceND, aI as bincount, b1 as broadcastTo, L as ceil, j as clipByValue, V as concat, a8 as conv1d, a7 as conv2d, a4 as conv2dTranspose, a2 as conv3d, K as cos, J as cosh, bv as cosineWindow, aJ as cumsum, be as customGrad, aH as denseBincount, b2 as depthToSpace, a3 as depthwiseConv2d, Y as dilation2d, c as div, d as divNoNan, bu as enclosingPowerOfTwo, ay as equal, I as erf, H as exp, b7 as expandDims, G as expm1, br as eye, b0 as fft, ai as fill, F as floor, f as floorDiv, aY as gather, aR as gatherND, bj as grad, bk as grads, aw as greater, av as greaterEqual, a$ as ifft, C as imag, al as image, aZ as irfft, au as less, at as lessEqual, bn as linalg, ah as linspace, aF as localResponseNormalization, E as log, D as log1p, aD as logSoftmax, bs as logSumExp, as as logicalAnd, ar as logicalNot, aq as logicalOr, bo as losses, aB as matMul, aQ as max, a0 as maxPool, Z as maxPool3d, $ as maxPoolWithArgmax, m as maximum, aP as mean, aO as min, a as minimum, b6 as mirrorPad, e as mod, ag as multinomial, B as neg, bt as norm, ax as notEqual, af as oneHot, ae as ones, ad as onesLike, b5 as pad, p as pow, i as prod, ac as randomUniform, ab as range, z as real, A as reciprocal, aX as reverse, a_ as rfft, y as round, r as rsqrt, aS as scatterND, x as selu, aj as setdiff1dAsync, w as sigmoid, u as sign, bq as signal, v as sin, q as sinh, X as slice, aE as softmax, o as softplus, b4 as spaceToBatchND, aC as sparseToDense, bp as spectral, aT as split, n as sqrt, l as square, s as squaredDifference, aV as squeeze, U as stack, aW as stridedSlice, b as sub, t as tan, k as tanh, ao as tensor1d, b9 as tensor2d, aU as tile, an as topk, aA as transpose, aa as truncatedNormal, am as unique, W as unstack, bl as valueAndGrad, bm as valueAndGrads, bh as variableGrads, ap as where, ak as whereAsync, a9 as zerosLike } from '../common/ops-5fc79cd7.js';
export { I as backend_util, N as gather_util, M as slice_util } from '../common/backend_util-5208330b.js';
import { s as scalar } from '../common/scalar-30b4cf58.js';
export { s as scalar, z as zeros } from '../common/scalar-30b4cf58.js';
import '../common/index-50ec6044.js';

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
const DEFAULT_FILE_NAME_PREFIX = 'model';
const DEFAULT_JSON_EXTENSION_NAME = '.json';
const DEFAULT_WEIGHT_DATA_EXTENSION_NAME = '.weights.bin';
function defer(f) {
    return new Promise(resolve => setTimeout(resolve)).then(f);
}
class BrowserDownloads {
    constructor(fileNamePrefix) {
        if (!env().getBool('IS_BROWSER')) {
            // TODO(cais): Provide info on what IOHandlers are available under the
            //   current environment.
            throw new Error('browserDownloads() cannot proceed because the current environment ' +
                'is not a browser.');
        }
        if (fileNamePrefix.startsWith(BrowserDownloads.URL_SCHEME)) {
            fileNamePrefix = fileNamePrefix.slice(BrowserDownloads.URL_SCHEME.length);
        }
        if (fileNamePrefix == null || fileNamePrefix.length === 0) {
            fileNamePrefix = DEFAULT_FILE_NAME_PREFIX;
        }
        this.modelTopologyFileName = fileNamePrefix + DEFAULT_JSON_EXTENSION_NAME;
        this.weightDataFileName =
            fileNamePrefix + DEFAULT_WEIGHT_DATA_EXTENSION_NAME;
    }
    async save(modelArtifacts) {
        if (typeof (document) === 'undefined') {
            throw new Error('Browser downloads are not supported in ' +
                'this environment since `document` is not present');
        }
        const weightsURL = window.URL.createObjectURL(new Blob([modelArtifacts.weightData], { type: 'application/octet-stream' }));
        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
            throw new Error('BrowserDownloads.save() does not support saving model topology ' +
                'in binary formats yet.');
        }
        else {
            const weightsManifest = [{
                    paths: ['./' + this.weightDataFileName],
                    weights: modelArtifacts.weightSpecs
                }];
            const modelTopologyAndWeightManifest = {
                modelTopology: modelArtifacts.modelTopology,
                format: modelArtifacts.format,
                generatedBy: modelArtifacts.generatedBy,
                convertedBy: modelArtifacts.convertedBy,
                weightsManifest
            };
            if (modelArtifacts.signature != null) {
                modelTopologyAndWeightManifest.signature = modelArtifacts.signature;
            }
            if (modelArtifacts.userDefinedMetadata != null) {
                modelTopologyAndWeightManifest.userDefinedMetadata =
                    modelArtifacts.userDefinedMetadata;
            }
            if (modelArtifacts.modelInitializer != null) {
                modelTopologyAndWeightManifest.modelInitializer =
                    modelArtifacts.modelInitializer;
            }
            const modelTopologyAndWeightManifestURL = window.URL.createObjectURL(new Blob([JSON.stringify(modelTopologyAndWeightManifest)], { type: 'application/json' }));
            // If anchor elements are not provided, create them without attaching them
            // to parents, so that the downloaded file names can be controlled.
            const jsonAnchor = this.jsonAnchor == null ? document.createElement('a') :
                this.jsonAnchor;
            jsonAnchor.download = this.modelTopologyFileName;
            jsonAnchor.href = modelTopologyAndWeightManifestURL;
            // Trigger downloads by evoking a click event on the download anchors.
            // When multiple downloads are started synchronously, Firefox will only
            // save the last one.
            await defer(() => jsonAnchor.dispatchEvent(new MouseEvent('click')));
            if (modelArtifacts.weightData != null) {
                const weightDataAnchor = this.weightDataAnchor == null ?
                    document.createElement('a') :
                    this.weightDataAnchor;
                weightDataAnchor.download = this.weightDataFileName;
                weightDataAnchor.href = weightsURL;
                await defer(() => weightDataAnchor.dispatchEvent(new MouseEvent('click')));
            }
            return { modelArtifactsInfo: getModelArtifactsInfoForJSON(modelArtifacts) };
        }
    }
}
BrowserDownloads.URL_SCHEME = 'downloads://';
class BrowserFiles {
    constructor(files) {
        if (files == null || files.length < 1) {
            throw new Error(`When calling browserFiles, at least 1 file is required, ` +
                `but received ${files}`);
        }
        this.files = files;
    }
    async load() {
        const jsonFile = this.files[0];
        const weightFiles = this.files.slice(1);
        return new Promise((resolve, reject) => {
            const jsonReader = new FileReader();
            jsonReader.onload = (event) => {
                // tslint:disable-next-line:no-any
                const modelJSON = JSON.parse(event.target.result);
                const modelTopology = modelJSON.modelTopology;
                if (modelTopology == null) {
                    reject(new Error(`modelTopology field is missing from file ${jsonFile.name}`));
                    return;
                }
                if (weightFiles.length === 0) {
                    resolve({ modelTopology });
                }
                const weightsManifest = modelJSON.weightsManifest;
                if (weightsManifest == null) {
                    reject(new Error(`weightManifest field is missing from file ${jsonFile.name}`));
                    return;
                }
                let pathToFile;
                try {
                    pathToFile =
                        this.checkManifestAndWeightFiles(weightsManifest, weightFiles);
                }
                catch (err) {
                    reject(err);
                    return;
                }
                const weightSpecs = [];
                const paths = [];
                const perFileBuffers = [];
                weightsManifest.forEach(weightsGroup => {
                    weightsGroup.paths.forEach(path => {
                        paths.push(path);
                        perFileBuffers.push(null);
                    });
                    weightSpecs.push(...weightsGroup.weights);
                });
                weightsManifest.forEach(weightsGroup => {
                    weightsGroup.paths.forEach(path => {
                        const weightFileReader = new FileReader();
                        weightFileReader.onload = (event) => {
                            // tslint:disable-next-line:no-any
                            const weightData = event.target.result;
                            const index = paths.indexOf(path);
                            perFileBuffers[index] = weightData;
                            if (perFileBuffers.indexOf(null) === -1) {
                                const result = {
                                    modelTopology,
                                    weightSpecs,
                                    weightData: concatenateArrayBuffers(perFileBuffers),
                                    format: modelJSON.format,
                                    generatedBy: modelJSON.generatedBy,
                                    convertedBy: modelJSON.convertedBy
                                };
                                if (modelJSON.signature != null) {
                                    result.signature = modelJSON.signature;
                                }
                                if (modelJSON.userDefinedMetadata != null) {
                                    result.userDefinedMetadata = modelJSON.userDefinedMetadata;
                                }
                                if (modelJSON.modelInitializer != null) {
                                    result.modelInitializer = modelJSON.modelInitializer;
                                }
                                resolve(result);
                            }
                        };
                        weightFileReader.onerror = error => reject(`Failed to weights data from file of path '${path}'.`);
                        weightFileReader.readAsArrayBuffer(pathToFile[path]);
                    });
                });
            };
            jsonReader.onerror = error => reject(`Failed to read model topology and weights manifest JSON ` +
                `from file '${jsonFile.name}'. BrowserFiles supports loading ` +
                `Keras-style tf.Model artifacts only.`);
            jsonReader.readAsText(jsonFile);
        });
    }
    /**
     * Check the compatibility between weights manifest and weight files.
     */
    checkManifestAndWeightFiles(manifest, files) {
        const basenames = [];
        const fileNames = files.map(file => basename(file.name));
        const pathToFile = {};
        for (const group of manifest) {
            group.paths.forEach(path => {
                const pathBasename = basename(path);
                if (basenames.indexOf(pathBasename) !== -1) {
                    throw new Error(`Duplicate file basename found in weights manifest: ` +
                        `'${pathBasename}'`);
                }
                basenames.push(pathBasename);
                if (fileNames.indexOf(pathBasename) === -1) {
                    throw new Error(`Weight file with basename '${pathBasename}' is not provided.`);
                }
                else {
                    pathToFile[path] = files[fileNames.indexOf(pathBasename)];
                }
            });
        }
        if (basenames.length !== files.length) {
            throw new Error(`Mismatch in the number of files in weights manifest ` +
                `(${basenames.length}) and the number of weight files provided ` +
                `(${files.length}).`);
        }
        return pathToFile;
    }
}
const browserDownloadsRouter = (url) => {
    if (!env().getBool('IS_BROWSER')) {
        return null;
    }
    else {
        if (!Array.isArray(url) && url.startsWith(BrowserDownloads.URL_SCHEME)) {
            return browserDownloads(url.slice(BrowserDownloads.URL_SCHEME.length));
        }
        else {
            return null;
        }
    }
};
IORouterRegistry.registerSaveRouter(browserDownloadsRouter);
/**
 * Creates an IOHandler that triggers file downloads from the browser.
 *
 * The returned `IOHandler` instance can be used as model exporting methods such
 * as `tf.Model.save` and supports only saving.
 *
 * ```js
 * const model = tf.sequential();
 * model.add(tf.layers.dense(
 *     {units: 1, inputShape: [10], activation: 'sigmoid'}));
 * const saveResult = await model.save('downloads://mymodel');
 * // This will trigger downloading of two files:
 * //   'mymodel.json' and 'mymodel.weights.bin'.
 * console.log(saveResult);
 * ```
 *
 * @param fileNamePrefix Prefix name of the files to be downloaded. For use with
 *   `tf.Model`, `fileNamePrefix` should follow either of the following two
 *   formats:
 *   1. `null` or `undefined`, in which case the default file
 *      names will be used:
 *      - 'model.json' for the JSON file containing the model topology and
 *        weights manifest.
 *      - 'model.weights.bin' for the binary file containing the binary weight
 *        values.
 *   2. A single string or an Array of a single string, as the file name prefix.
 *      For example, if `'foo'` is provided, the downloaded JSON
 *      file and binary weights file will be named 'foo.json' and
 *      'foo.weights.bin', respectively.
 * @param config Additional configuration for triggering downloads.
 * @returns An instance of `BrowserDownloads` `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
function browserDownloads(fileNamePrefix = 'model') {
    return new BrowserDownloads(fileNamePrefix);
}
/**
 * Creates an IOHandler that loads model artifacts from user-selected files.
 *
 * This method can be used for loading from files such as user-selected files
 * in the browser.
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * // Note: This code snippet won't run properly without the actual file input
 * //   elements in the HTML DOM.
 *
 * // Suppose there are two HTML file input (`<input type="file" ...>`)
 * // elements.
 * const uploadJSONInput = document.getElementById('upload-json');
 * const uploadWeightsInput = document.getElementById('upload-weights');
 * const model = await tf.loadLayersModel(tf.io.browserFiles(
 *     [uploadJSONInput.files[0], uploadWeightsInput.files[0]]));
 * ```
 *
 * @param files `File`s to load from. Currently, this function supports only
 *   loading from files that contain Keras-style models (i.e., `tf.Model`s), for
 *   which an `Array` of `File`s is expected (in that order):
 *   - A JSON file containing the model topology and weight manifest.
 *   - Optionally, One or more binary files containing the binary weights.
 *     These files must have names that match the paths in the `weightsManifest`
 *     contained by the aforementioned JSON file, or errors will be thrown
 *     during loading. These weights files have the same format as the ones
 *     generated by `tensorflowjs_converter` that comes with the `tensorflowjs`
 *     Python PIP package. If no weights files are provided, only the model
 *     topology will be loaded from the JSON file above.
 * @returns An instance of `Files` `IOHandler`.
 *
 * @doc {
 *   heading: 'Models',
 *   subheading: 'Loading',
 *   namespace: 'io',
 *   ignoreCI: true
 * }
 */
function browserFiles(files) {
    return new BrowserFiles(files);
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
class PassthroughLoader {
    constructor(modelArtifacts) {
        this.modelArtifacts = modelArtifacts;
    }
    async load() {
        return this.modelArtifacts;
    }
}
class PassthroughSaver {
    constructor(saveHandler) {
        this.saveHandler = saveHandler;
    }
    async save(modelArtifacts) {
        return this.saveHandler(modelArtifacts);
    }
}
/**
 * Creates an IOHandler that loads model artifacts from memory.
 *
 * When used in conjunction with `tf.loadLayersModel`, an instance of
 * `tf.LayersModel` (Keras-style) can be constructed from the loaded artifacts.
 *
 * ```js
 * const model = await tf.loadLayersModel(tf.io.fromMemory(
 *     modelTopology, weightSpecs, weightData));
 * ```
 *
 * @param modelArtifacts a object containing model topology (i.e., parsed from
 *   the JSON format).
 * @param weightSpecs An array of `WeightsManifestEntry` objects describing the
 *   names, shapes, types, and quantization of the weight data.
 * @param weightData A single `ArrayBuffer` containing the weight data,
 *   concatenated in the order described by the weightSpecs.
 * @param trainingConfig Model training configuration. Optional.
 *
 * @returns A passthrough `IOHandler` that simply loads the provided data.
 */
function fromMemory(modelArtifacts, weightSpecs, weightData, trainingConfig) {
    if (arguments.length === 1) {
        const isModelArtifacts = modelArtifacts.modelTopology != null ||
            modelArtifacts.weightSpecs != null;
        if (isModelArtifacts) {
            return new PassthroughLoader(modelArtifacts);
        }
        else {
            // Legacy support: with only modelTopology.
            // TODO(cais): Remove this deprecated API.
            console.warn('Please call tf.io.fromMemory() with only one argument. ' +
                'The argument should be of type ModelArtifacts. ' +
                'The multi-argument signature of tf.io.fromMemory() has been ' +
                'deprecated and will be removed in a future release.');
            return new PassthroughLoader({ modelTopology: modelArtifacts });
        }
    }
    else {
        // Legacy support.
        // TODO(cais): Remove this deprecated API.
        console.warn('Please call tf.io.fromMemory() with only one argument. ' +
            'The argument should be of type ModelArtifacts. ' +
            'The multi-argument signature of tf.io.fromMemory() has been ' +
            'deprecated and will be removed in a future release.');
        return new PassthroughLoader({
            modelTopology: modelArtifacts,
            weightSpecs,
            weightData,
            trainingConfig
        });
    }
}
/**
 * Creates an IOHandler that passes saved model artifacts to a callback.
 *
 * ```js
 * function handleSave(artifacts) {
 *   // ... do something with the artifacts ...
 *   return {modelArtifactsInfo: {...}, ...};
 * }
 *
 * const saveResult = model.save(tf.io.withSaveHandler(handleSave));
 * ```
 *
 * @param saveHandler A function that accepts a `ModelArtifacts` and returns a
 *     `SaveResult`.
 */
function withSaveHandler(saveHandler) {
    return new PassthroughSaver(saveHandler);
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

var io = /*#__PURE__*/Object.freeze({
    __proto__: null,
    browserFiles: browserFiles,
    browserHTTPRequest: browserHTTPRequest,
    concatenateArrayBuffers: concatenateArrayBuffers,
    decodeWeights: decodeWeights,
    encodeWeights: encodeWeights,
    fromMemory: fromMemory,
    getLoadHandlers: getLoadHandlers,
    getModelArtifactsInfoForJSON: getModelArtifactsInfoForJSON,
    getSaveHandlers: getSaveHandlers,
    http: http,
    isHTTPScheme: isHTTPScheme,
    loadWeights: loadWeights,
    registerLoadRouter: registerLoadRouter,
    registerSaveRouter: registerSaveRouter,
    weightsLoaderFactory: weightsLoaderFactory,
    withSaveHandler: withSaveHandler,
    copyModel: copyModel,
    listModels: listModels,
    moveModel: moveModel,
    removeModel: removeModel
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
 * Computes the confusion matrix from true labels and predicted labels.
 *
 * ```js
 * const labels = tf.tensor1d([0, 1, 2, 1, 0], 'int32');
 * const predictions = tf.tensor1d([0, 2, 2, 1, 0], 'int32');
 * const numClasses = 3;
 * const out = tf.math.confusionMatrix(labels, predictions, numClasses);
 * out.print();
 * // Expected output matrix:
 * // [[2, 0, 0],
 * //  [0, 1, 1],
 * //  [0, 0, 1]]
 * ```
 *
 * @param labels The target labels, assumed to be 0-based integers
 *   for the classes. The shape is `[numExamples]`, where
 *   `numExamples` is the number of examples included.
 * @param predictions The predicted classes, assumed to be
 *   0-based integers for the classes. Must have the same shape as `labels`.
 * @param numClasses Number of all classes, as an integer.
 *   Its value must be larger than the largest element in `labels` and
 *   `predictions`.
 * @returns The confusion matrix as a int32-type 2D tensor. The value at
 *   row `r` and column `c` is the number of times examples of actual class
 *   `r` were predicted as class `c`.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function confusionMatrix_(labels, predictions, numClasses) {
    const $labels = convertToTensor(labels, 'labels', 'confusionMatrix');
    const $predictions = convertToTensor(predictions, 'predictions', 'confusionMatrix');
    assert(numClasses == null || numClasses > 0 && Number.isInteger(numClasses), () => `If provided, numClasses must be a positive integer, ` +
        `but got ${numClasses}`);
    assert($labels.rank === 1, () => `Expected the rank of labels to be 1, but got ${$labels.rank}`);
    assert($predictions.rank === 1, () => `Expected the rank of predictions to be 1, ` +
        `but got ${$predictions.rank}`);
    assert($labels.shape[0] === $predictions.shape[0], () => `Mismatch in the number of examples: ` +
        `${$labels.shape[0]} vs. ${$predictions.shape[0]}. ` +
        `Labels and predictions should have the same number of elements.`);
    assert(numClasses > 0 && Number.isInteger(numClasses), () => `numClasses is required to be a positive integer, but got ` +
        `${numClasses}`);
    // TODO(cais): In the future, if oneHot supports tensors inputs for
    //   `numClasses`, `confusionMatrix` can make `numClasses` optional.
    const oneHotLabels = oneHot(cast($labels, 'int32'), numClasses);
    const oneHotPredictions = oneHot(cast($predictions, 'int32'), numClasses);
    const oneHotLabelsT = transpose(oneHotLabels);
    const product = matMul(oneHotLabelsT, oneHotPredictions);
    return cast(product, 'int32');
}
const confusionMatrix = op({ confusionMatrix_ });

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

var math = /*#__PURE__*/Object.freeze({
    __proto__: null,
    confusionMatrix: confusionMatrix
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
 * Creates rank-3 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor3d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor3d([[[1], [2]], [[3], [4]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. If not provided,  it is inferred from
 *     `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor3d(values, shape, dtype) {
    assertNonNull(values);
    if (shape != null && shape.length !== 3) {
        throw new Error('tensor3d() requires shape to have three numbers');
    }
    const inferredShape = inferShape(values, dtype);
    if (inferredShape.length !== 3 && inferredShape.length !== 1) {
        throw new Error('tensor3d() requires values to be number[][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor3d() requires shape to be provided when `values` ' +
            'are a flat array');
    }
    return makeTensor(values, shape, inferredShape, dtype);
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
let fromPixels2DContext;
/**
 * Creates a `tf.Tensor` from an image.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * tf.browser.fromPixels(image).print();
 * ```
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
function fromPixels_(pixels, numChannels = 3) {
    // Sanity checks.
    if (numChannels > 4) {
        throw new Error('Cannot construct Tensor with more than 4 channels from pixels.');
    }
    if (pixels == null) {
        throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
    }
    let isPixelData = false;
    let isImageData = false;
    let isVideo = false;
    let isImage = false;
    let isCanvasLike = false;
    let isImageBitmap = false;
    if (pixels.data instanceof Uint8Array) {
        isPixelData = true;
    }
    else if (typeof (ImageData) !== 'undefined' && pixels instanceof ImageData) {
        isImageData = true;
    }
    else if (typeof (HTMLVideoElement) !== 'undefined' &&
        pixels instanceof HTMLVideoElement) {
        isVideo = true;
    }
    else if (typeof (HTMLImageElement) !== 'undefined' &&
        pixels instanceof HTMLImageElement) {
        isImage = true;
        // tslint:disable-next-line: no-any
    }
    else if (pixels.getContext != null) {
        isCanvasLike = true;
    }
    else if (typeof (ImageBitmap) !== 'undefined' &&
        pixels instanceof ImageBitmap) {
        isImageBitmap = true;
    }
    else {
        throw new Error('pixels passed to tf.browser.fromPixels() must be either an ' +
            `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
            `in browser, or OffscreenCanvas, ImageData in webworker` +
            ` or {data: Uint32Array, width: number, height: number}, ` +
            `but was ${pixels.constructor.name}`);
    }
    if (isVideo) {
        const HAVE_CURRENT_DATA_READY_STATE = 2;
        if (isVideo &&
            pixels.readyState <
                HAVE_CURRENT_DATA_READY_STATE) {
            throw new Error('The video element has not loaded data yet. Please wait for ' +
                '`loadeddata` event on the <video> element.');
        }
    }
    // If the current backend has 'FromPixels' registered, it has a more
    // efficient way of handling pixel uploads, so we call that.
    const kernel = getKernel(FromPixels, ENGINE.backendName);
    if (kernel != null) {
        const inputs = { pixels };
        const attrs = { numChannels };
        return ENGINE.runKernel(FromPixels, inputs, attrs);
    }
    const [width, height] = isVideo ?
        [
            pixels.videoWidth,
            pixels.videoHeight
        ] :
        [pixels.width, pixels.height];
    let vals;
    if (isCanvasLike) {
        vals =
            // tslint:disable-next-line:no-any
            pixels.getContext('2d').getImageData(0, 0, width, height).data;
    }
    else if (isImageData || isPixelData) {
        vals = pixels.data;
    }
    else if (isImage || isVideo || isImageBitmap) {
        if (fromPixels2DContext == null) {
            fromPixels2DContext = document.createElement('canvas').getContext('2d');
        }
        fromPixels2DContext.canvas.width = width;
        fromPixels2DContext.canvas.height = height;
        fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
        vals = fromPixels2DContext.getImageData(0, 0, width, height).data;
    }
    let values;
    if (numChannels === 4) {
        values = new Int32Array(vals);
    }
    else {
        const numPixels = width * height;
        values = new Int32Array(numPixels * numChannels);
        for (let i = 0; i < numPixels; i++) {
            for (let channel = 0; channel < numChannels; ++channel) {
                values[i * numChannels + channel] = vals[i * 4 + channel];
            }
        }
    }
    const outShape = [height, width, numChannels];
    return tensor3d(values, outShape, 'int32');
}
/**
 * Draws a `tf.Tensor` of pixel values to a byte array or optionally a
 * canvas.
 *
 * When the dtype of the input is 'float32', we assume values in the range
 * [0-1]. Otherwise, when input is 'int32', we assume values in the range
 * [0-255].
 *
 * Returns a promise that resolves when the canvas has been drawn to.
 *
 * @param img A rank-2 or rank-3 tensor. If rank-2, draws grayscale. If
 *     rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
 * grayscale. When depth of 3, we draw with the first three components of
 * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
 * 4, all four components of the depth dimension correspond to r, g, b, a.
 * @param canvas The canvas to draw to.
 *
 * @doc {heading: 'Browser', namespace: 'browser'}
 */
async function toPixels(img, canvas) {
    let $img = convertToTensor(img, 'img', 'toPixels');
    if (!(img instanceof Tensor)) {
        // Assume int32 if user passed a native array.
        const originalImgTensor = $img;
        $img = cast(originalImgTensor, 'int32');
        originalImgTensor.dispose();
    }
    if ($img.rank !== 2 && $img.rank !== 3) {
        throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${$img.rank}.`);
    }
    const [height, width] = $img.shape.slice(0, 2);
    const depth = $img.rank === 2 ? 1 : $img.shape[2];
    if (depth > 4 || depth === 2) {
        throw new Error(`toPixels only supports depth of size ` +
            `1, 3 or 4 but got ${depth}`);
    }
    if ($img.dtype !== 'float32' && $img.dtype !== 'int32') {
        throw new Error(`Unsupported type for toPixels: ${$img.dtype}.` +
            ` Please use float32 or int32 tensors.`);
    }
    const data = await $img.data();
    const multiplier = $img.dtype === 'float32' ? 255 : 1;
    const bytes = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < height * width; ++i) {
        const rgba = [0, 0, 0, 255];
        for (let d = 0; d < depth; d++) {
            const value = data[i * depth + d];
            if ($img.dtype === 'float32') {
                if (value < 0 || value > 1) {
                    throw new Error(`Tensor values for a float32 Tensor must be in the ` +
                        `range [0 - 1] but encountered ${value}.`);
                }
            }
            else if ($img.dtype === 'int32') {
                if (value < 0 || value > 255) {
                    throw new Error(`Tensor values for a int32 Tensor must be in the ` +
                        `range [0 - 255] but encountered ${value}.`);
                }
            }
            if (depth === 1) {
                rgba[0] = value * multiplier;
                rgba[1] = value * multiplier;
                rgba[2] = value * multiplier;
            }
            else {
                rgba[d] = value * multiplier;
            }
        }
        const j = i * 4;
        bytes[j + 0] = Math.round(rgba[0]);
        bytes[j + 1] = Math.round(rgba[1]);
        bytes[j + 2] = Math.round(rgba[2]);
        bytes[j + 3] = Math.round(rgba[3]);
    }
    if (canvas != null) {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = new ImageData(bytes, width, height);
        ctx.putImageData(imageData, 0, 0);
    }
    if ($img !== img) {
        $img.dispose();
    }
    return bytes;
}
const fromPixels = op({ fromPixels_ });

var browser = /*#__PURE__*/Object.freeze({
    __proto__: null,
    toPixels: toPixels,
    fromPixels: fromPixels
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
 * Serializable defines the serialization contract.
 *
 * TFJS requires serializable classes to return their className when asked
 * to avoid issues with minification.
 */
class Serializable {
    /**
     * Return the class name for this class to use in serialization contexts.
     *
     * Generally speaking this will be the same thing that constructor.name
     * would have returned.  However, the class name needs to be robust
     * against minification for serialization/deserialization to work properly.
     *
     * There's also places such as initializers.VarianceScaling, where
     * implementation details between different languages led to different
     * class hierarchies and a non-leaf node is used for serialization purposes.
     */
    getClassName() {
        return this.constructor
            .className;
    }
    /**
     * Creates an instance of T from a ConfigDict.
     *
     * This works for most descendants of serializable.  A few need to
     * provide special handling.
     * @param cls A Constructor for the class to instantiate.
     * @param config The Configuration for the object.
     */
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config);
    }
}
/**
 * Maps string keys to class constructors.
 *
 * Used during (de)serialization from the cross-language JSON format, which
 * requires the class name in the serialization format matches the class
 * names as used in Python, should it exist.
 */
class SerializationMap {
    constructor() {
        this.classNameMap = {};
    }
    /**
     * Returns the singleton instance of the map.
     */
    static getMap() {
        if (SerializationMap.instance == null) {
            SerializationMap.instance = new SerializationMap();
        }
        return SerializationMap.instance;
    }
    /**
     * Registers the class as serializable.
     */
    static register(cls) {
        SerializationMap.getMap().classNameMap[cls.className] =
            [cls, cls.fromConfig];
    }
}
/**
 * Register a class with the serialization map of TensorFlow.js.
 *
 * This is often used for registering custom Layers, so they can be
 * serialized and deserialized.
 *
 * Example:
 *
 * ```js
 * class MyCustomLayer extends tf.layers.Layer {
 *   static className = 'MyCustomLayer';
 *
 *   constructor(config) {
 *     super(config);
 *   }
 * }
 * tf.serialization.registerClass(MyCustomLayer);
 * ```
 *
 * @param cls The class to be registered. It must have a public static member
 *   called `className` defined and the value must be a non-empty string.
 *
 * @doc {heading: 'Models', subheading: 'Serialization', ignoreCI: true}
 */
function registerClass(cls) {
    assert(cls.className != null, () => `Class being registered does not have the static className ` +
        `property defined.`);
    assert(typeof cls.className === 'string', () => `className is required to be a string, but got type ` +
        typeof cls.className);
    assert(cls.className.length > 0, () => `Class being registered has an empty-string as its className, ` +
        `which is disallowed.`);
    SerializationMap.register(cls);
}

var serialization = /*#__PURE__*/Object.freeze({
    __proto__: null,
    Serializable: Serializable,
    SerializationMap: SerializationMap,
    registerClass: registerClass
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
const TEST_EPSILON_FLOAT32 = 1e-3;
const TEST_EPSILON_FLOAT16 = 1e-1;
function expectArraysClose(actual, expected, epsilon) {
    if (epsilon == null) {
        epsilon = testEpsilon();
    }
    return expectArraysPredicate(actual, expected, (a, b) => areClose(a, b, epsilon));
}
function testEpsilon() {
    return ENGINE.backend.floatPrecision() === 32 ? TEST_EPSILON_FLOAT32 :
        TEST_EPSILON_FLOAT16;
}
function expectArraysPredicate(actual, expected, predicate) {
    let checkClassType = true;
    if (isTypedArray(actual) || isTypedArray(expected)) {
        checkClassType = false;
    }
    if (isTypedArray(actual) && isTypedArray(expected)) {
        checkClassType = true;
    }
    if (checkClassType) {
        const aType = actual.constructor.name;
        const bType = expected.constructor.name;
        if (aType !== bType) {
            throw new Error(`Arrays are of different type. Actual: ${aType}. ` +
                `Expected: ${bType}`);
        }
    }
    if (Array.isArray(actual) && Array.isArray(expected)) {
        const actualShape = inferShape(actual);
        const expectedShape = inferShape(expected);
        if (!arraysEqual(actualShape, expectedShape)) {
            throw new Error(`Arrays have different shapes. ` +
                `Actual: [${actualShape}]. Expected: [${expectedShape}]`);
        }
    }
    const actualFlat = isTypedArray(actual) ? actual : flatten(actual);
    const expectedFlat = isTypedArray(expected) ?
        expected :
        flatten(expected);
    if (actualFlat.length !== expectedFlat.length) {
        throw new Error(`Arrays have different lengths actual: ${actualFlat.length} vs ` +
            `expected: ${expectedFlat.length}.\n` +
            `Actual:   ${actualFlat}.\n` +
            `Expected: ${expectedFlat}.`);
    }
    for (let i = 0; i < expectedFlat.length; ++i) {
        const a = actualFlat[i];
        const e = expectedFlat[i];
        if (!predicate(a, e)) {
            throw new Error(`Arrays differ: actual[${i}] = ${a}, expected[${i}] = ${e}.\n` +
                `Actual:   ${actualFlat}.\n` +
                `Expected: ${expectedFlat}.`);
        }
    }
}
function expectPromiseToFail(fn, done) {
    fn().then(() => done.fail(), () => done());
}
function expectArraysEqual(actual, expected) {
    const exp = typeof expected === 'string' || typeof expected === 'number' ||
        typeof expected === 'boolean' ?
        [expected] :
        expected;
    if (isString(actual) || isString(actual[0]) ||
        isString(expected) || isString(expected[0])) {
        // tslint:disable-next-line: triple-equals
        return expectArraysPredicate(actual, exp, (a, b) => a == b);
    }
    return expectArraysPredicate(actual, expected, (a, b) => areClose(a, b, 0));
}
function expectNumbersClose(a, e, epsilon) {
    if (epsilon == null) {
        epsilon = testEpsilon();
    }
    if (!areClose(a, e, epsilon)) {
        throw new Error(`Numbers differ: actual === ${a}, expected === ${e}`);
    }
}
function areClose(a, e, epsilon) {
    if (!isFinite(a) && !isFinite(e)) {
        return true;
    }
    if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
        return false;
    }
    return true;
}
function expectValuesInRange(actual, low, high) {
    for (let i = 0; i < actual.length; i++) {
        if (actual[i] < low || actual[i] > high) {
            throw new Error(`Value out of range:${actual[i]} low: ${low}, high: ${high}`);
        }
    }
}
function expectArrayBuffersEqual(actual, expected) {
    // Safari & Jasmine don't like comparing ArrayBuffers directly. Wrapping in
    // a Float32Array solves this issue.
    expect(new Float32Array(actual)).toEqual(new Float32Array(expected));
}
/** Encodes strings into utf-8 bytes. */
function encodeStrings(a) {
    for (let i = 0; i < a.length; i++) {
        const val = a[i];
        if (Array.isArray(val)) {
            encodeStrings(val);
        }
        else {
            a[i] = encodeString(val);
        }
    }
    return a;
}

var test_util = /*#__PURE__*/Object.freeze({
    __proto__: null,
    TEST_EPSILON_FLOAT16: TEST_EPSILON_FLOAT16,
    expectArraysClose: expectArraysClose,
    testEpsilon: testEpsilon,
    expectPromiseToFail: expectPromiseToFail,
    expectArraysEqual: expectArraysEqual,
    expectNumbersClose: expectNumbersClose,
    expectValuesInRange: expectValuesInRange,
    expectArrayBuffersEqual: expectArrayBuffersEqual,
    encodeStrings: encodeStrings
});

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
/**
 * Computes the next state and output of a BasicLSTMCell.
 *
 * Returns `[newC, newH]`.
 *
 * Derived from tf.contrib.rnn.BasicLSTMCell.
 *
 * @param forgetBias Forget bias for the cell.
 * @param lstmKernel The weights for the cell.
 * @param lstmBias The bias for the cell.
 * @param data The input to the cell.
 * @param c Previous cell state.
 * @param h Previous cell output.
 *
 * @doc {heading: 'Operations', subheading: 'RNN'}
 */
function basicLSTMCell_(forgetBias, lstmKernel, lstmBias, data, c, h) {
    const $forgetBias = convertToTensor(forgetBias, 'forgetBias', 'basicLSTMCell');
    const $lstmKernel = convertToTensor(lstmKernel, 'lstmKernel', 'basicLSTMCell');
    const $lstmBias = convertToTensor(lstmBias, 'lstmBias', 'basicLSTMCell');
    const $data = convertToTensor(data, 'data', 'basicLSTMCell');
    const $c = convertToTensor(c, 'c', 'basicLSTMCell');
    const $h = convertToTensor(h, 'h', 'basicLSTMCell');
    const combined = concat([$data, $h], 1);
    const weighted = matMul(combined, $lstmKernel);
    const res = add(weighted, $lstmBias);
    // i = input_gate, j = new_input, f = forget_gate, o = output_gate
    const batchSize = res.shape[0];
    const sliceCols = res.shape[1] / 4;
    const sliceSize = [batchSize, sliceCols];
    const i = slice(res, [0, 0], sliceSize);
    const j = slice(res, [0, sliceCols], sliceSize);
    const f = slice(res, [0, sliceCols * 2], sliceSize);
    const o = slice(res, [0, sliceCols * 3], sliceSize);
    const newC = add(mul(sigmoid(i), tanh(j)), mul($c, sigmoid(add($forgetBias, f))));
    const newH = mul(tanh(newC), sigmoid(o));
    return [newC, newH];
}
const basicLSTMCell = op({ basicLSTMCell_ });

/**
 * Batch normalization, strictly for 2D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm2d_(x, mean, variance, offset, scale, varianceEpsilon) {
    const $x = convertToTensor(x, 'x', 'batchNorm');
    const $mean = convertToTensor(mean, 'mean', 'batchNorm');
    const $variance = convertToTensor(variance, 'variance', 'batchNorm');
    let $scale;
    if (scale != null) {
        $scale = convertToTensor(scale, 'scale', 'batchNorm');
    }
    let $offset;
    if (offset != null) {
        $offset = convertToTensor(offset, 'offset', 'batchNorm');
    }
    assert($x.rank === 2, () => `Error in batchNorm2D: x must be rank 2 but got rank ` +
        `${$x.rank}.`);
    assert($mean.rank === 2 || $mean.rank === 1, () => `Error in batchNorm2D: mean must be rank 2 or rank 1 but ` +
        `got rank ${$mean.rank}.`);
    assert($variance.rank === 2 || $variance.rank === 1, () => `Error in batchNorm2D: variance must be rank 2 or rank 1 ` +
        `but got rank ${$variance.rank}.`);
    if ($scale != null) {
        assert($scale.rank === 2 || $scale.rank === 1, () => `Error in batchNorm2D: scale must be rank 2 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
    }
    if ($offset != null) {
        assert($offset.rank === 2 || $offset.rank === 1, () => `Error in batchNorm2D: offset must be rank 2 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
    }
    return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}
const batchNorm2d = op({ batchNorm2d_ });

/**
 * Batch normalization, strictly for 3D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm3d_(x, mean, variance, offset, scale, varianceEpsilon) {
    const $x = convertToTensor(x, 'x', 'batchNorm');
    const $mean = convertToTensor(mean, 'mean', 'batchNorm');
    const $variance = convertToTensor(variance, 'variance', 'batchNorm');
    let $scale;
    if (scale != null) {
        $scale = convertToTensor(scale, 'scale', 'batchNorm');
    }
    let $offset;
    if (offset != null) {
        $offset = convertToTensor(offset, 'offset', 'batchNorm');
    }
    assert($x.rank === 3, () => `Error in batchNorm3D: x must be rank 3 but got rank ` +
        `${$x.rank}.`);
    assert($mean.rank === 3 || $mean.rank === 1, () => `Error in batchNorm3D: mean must be rank 3 or rank 1 but ` +
        `got rank ${$mean.rank}.`);
    assert($variance.rank === 3 || $variance.rank === 1, () => `Error in batchNorm3D: variance must be rank 3 or rank 1 ` +
        `but got rank ${$variance.rank}.`);
    if ($scale != null) {
        assert($scale.rank === 3 || $scale.rank === 1, () => `Error in batchNorm3D: scale must be rank 3 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
    }
    if ($offset != null) {
        assert($offset.rank === 3 || $offset.rank === 1, () => `Error in batchNorm3D: offset must be rank 3 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
    }
    return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}
const batchNorm3d = op({ batchNorm3d_ });

/**
 * Batch normalization, strictly for 4D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm4d_(x, mean, variance, offset, scale, varianceEpsilon) {
    const $x = convertToTensor(x, 'x', 'batchNorm');
    const $mean = convertToTensor(mean, 'mean', 'batchNorm');
    const $variance = convertToTensor(variance, 'variance', 'batchNorm');
    let $scale;
    if (scale != null) {
        $scale = convertToTensor(scale, 'scale', 'batchNorm');
    }
    let $offset;
    if (offset != null) {
        $offset = convertToTensor(offset, 'offset', 'batchNorm');
    }
    assert($x.rank === 4, () => `Error in batchNorm4D: x must be rank 4 but got rank ` +
        `${$x.rank}.`);
    assert($mean.rank === 4 || $mean.rank === 1, () => `Error in batchNorm4D: mean must be rank 4 or rank 1 but ` +
        `got rank ${$mean.rank}.`);
    assert($variance.rank === 4 || $variance.rank === 1, () => `Error in batchNorm4D: variance must be rank 4 or rank 1 ` +
        `but got rank ${$variance.rank}.`);
    if ($scale != null) {
        assert($scale.rank === 4 || $scale.rank === 1, () => `Error in batchNorm4D: scale must be rank 4 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
    }
    if ($offset != null) {
        assert($offset.rank === 4 || $offset.rank === 1, () => `Error in batchNorm4D: offset must be rank 4 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
    }
    return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}
const batchNorm4d = op({ batchNorm4d_ });

/**
 * Concatenates a list of`tf.Tensor1D`s along an axis. See `concat` for details.
 *
 * For example, if:
 * A: shape(3) = |r1, g1, b1|
 * B: shape(2) = |r2, g2|
 * C = tf.concat1d([A, B]) == |r1, g1, b1, r2, g2|
 *
 * @param tensors A list of`tf.Tensor`s to concatenate.
 * @return The concatenated array.
 */
function concat1d_(tensors) {
    return concat(tensors, 0 /* axis */);
}
const concat1d = op({ concat1d_ });

/**
 * Concatenates a list of`tf.Tensor2D`s along an axis. See `concat` for details.
 *
 * For example, if:
 * A: shape(2, 3) = | r1, g1, b1 |
 *                  | r2, g2, b2 |
 *
 * B: shape(2, 3) = | r3, g3, b3 |
 *                  | r4, g4, b4 |
 *
 * C = tf.concat2d([A, B], axis)
 *
 * if axis = 0:
 * C: shape(4, 3) = | r1, g1, b1 |
 *                  | r2, g2, b2 |
 *                  | r3, g3, b3 |
 *                  | r4, g4, b4 |
 *
 * if axis = 1:
 * C = shape(2, 6) = | r1, g1, b1, r3, g3, b3 |
 *                   | r2, g2, b2, r4, g4, b4 |
 *
 *
 * @param tensors A list of `tf.Tensor`s to concatenate.
 * @param axis The axis to concatenate along.
 * @return The concatenated array.
 */
function concat2d_(tensors, axis) {
    return concat(tensors, axis);
}
const concat2d = op({ concat2d_ });

/**
 * Concatenates a list of `tf.Tensor3D`s along an axis.
 * See `concat` for details.
 *
 * For example, if:
 * A: shape(2, 1, 3) = | r1, g1, b1 |
 *                     | r2, g2, b2 |
 *
 * B: shape(2, 1, 3) = | r3, g3, b3 |
 *                     | r4, g4, b4 |
 *
 * C = tf.concat3d([A, B], axis)
 *
 * if axis = 0:
 * C: shape(4, 1, 3) = | r1, g1, b1 |
 *                     | r2, g2, b2 |
 *                     | r3, g3, b3 |
 *                     | r4, g4, b4 |
 *
 * if axis = 1:
 * C: shape(2, 2, 3) = | r1, g1, b1, r3, g3, b3 |
 *                     | r2, g2, b2, r4, g4, b4 |
 *
 * if axis = 2:
 * C = shape(2, 1, 6) = | r1, g1, b1, r3, g3, b3 |
 *                      | r2, g2, b2, r4, g4, b4 |
 *
 * @param tensors A list of`tf.Tensor`s to concatenate.
 * @param axis The axis to concate along.
 * @return The concatenated array.
 */
function concat3d_(tensors, axis) {
    return concat(tensors, axis);
}
const concat3d = op({ concat3d_ });

/**
 * Concatenates a list of `tf.Tensor4D`s along an axis.
 * See `concat` for details.
 *
 * @param tensors A list of `tf.Tensor`s to concatenate.
 * @param axis The axis to concate along.
 * @return The concatenated array.
 */
function concat4d_(tensors, axis) {
    return concat(tensors, axis);
}
const concat4d = op({ concat4d_ });

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
 * Computes the derivative of the input of a 3D convolution.
 *
 * @param xShape The shape of the input: [batch, depth, height, width,
 * in_channels]. If length of 4, batch of 1 is assumed.
 * @param dy The derivative of the output, of rank 5 or rank 4 of shape
 *   `[batch, outDepth, outHeight, outWidth, in_channels]`.
 * If rank 4, batch of 1 is assumed.
 * @param filter The filter, rank 5, of shape
 *     `[filterDepth, filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideDepth, strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm used:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 */
function conv3DBackpropInput_(xShape, dy, filter, strides, pad) {
    assert(xShape.length === dy.rank, () => `Length of inShape ` +
        `(${xShape.length}) and rank of dy (${dy.rank}) must match`);
    let xShape5D = xShape;
    let dy5D = dy;
    let reshapedTo5D = false;
    if (dy.rank === 4) {
        reshapedTo5D = true;
        dy5D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]]);
        xShape5D = [1, xShape[0], xShape[1], xShape[2], xShape[3]];
    }
    const inDepth = xShape5D[4];
    const outDepth = dy5D.shape[4];
    assert(xShape5D.length === 5, () => `Error in conv3dDerInput: inShape must be length 5, but got length ` +
        `${xShape5D.length}.`);
    assert(dy5D.rank === 5, () => `Error in conv3dDerInput: dy must be rank 5, but got ` +
        `rank ${dy5D.rank}`);
    assert(filter.rank === 5, () => `Error in conv3dDerInput: filter must be rank 5, but got ` +
        `rank ${filter.rank}`);
    assert(inDepth === filter.shape[3], () => `Error in conv3dDerInput: depth of input (${inDepth}) must ` +
        `match input depth for filter ${filter.shape[3]}.`);
    assert(outDepth === filter.shape[4], () => `Error in conv3dDerInput: depth of output (${outDepth}) must ` +
        `match output depth for filter ${filter.shape[4]}.`);
    const inputs = { dy: dy5D, filter };
    const attrs = { pad, strides, inputShape: xShape5D };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(Conv3DBackpropInputV2, inputs, attrs);
    if (reshapedTo5D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]);
    }
    return res;
}
const conv3DBackpropInput = op({ conv3DBackpropInput_ });

/**
 * Computes the transposed 3D convolution of a volume, also known as a
 * deconvolution.
 *
 * @param x The input image, of rank 5 or rank 4, of shape
 *   `[batch, depth, height, width, inDepth]`. If rank 4, batch of 1 is assumed.
 * @param filter The filter, rank 4, of shape
 *     `[depth, filterHeight, filterWidth, outDepth, inDepth]`.
 *     `inDepth` must match `inDepth` in `x`.
 * @param outputShape Output shape, of rank 5 or rank 4:
 *     `[batch, depth, height, width, outDepth]`. If rank 3, batch of 1 is
 *    assumed.
 * @param strides The strides of the original convolution:
 *     `[strideDepth, strideHeight, strideWidth]`.
 * @param pad  The type of padding algorithm used in the non-transpose version
 *    of the op.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function conv3dTranspose_(x, filter, outputShape, strides, pad) {
    const $x = convertToTensor(x, 'x', 'conv3dTranspose');
    const $filter = convertToTensor(filter, 'filter', 'conv3dTranspose');
    return conv3DBackpropInput(outputShape, $x, $filter, strides, pad);
}
const conv3dTranspose = op({ conv3dTranspose_ });

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
 * Returns a diagonal tensor with a given diagonal values.
 *
 * Given a diagonal, this operation returns a tensor with the diagonal and
 * everything else padded with zeros.
 *
 * Assume the input has dimensions `[D1,..., Dk]`, then the output is a tensor
 * of rank 2k with dimensions `[D1,..., Dk, D1,..., Dk]`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * tf.diag(x).print()
 * ```
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4, 5, 6, 6, 8], [4, 2])
 *
 * tf.diag(x).print()
 * ```
 * @param x The input tensor.
 */
function diag_(x) {
    const $x = convertToTensor(x, 'x', 'diag');
    const inputs = { x: $x };
    return ENGINE.runKernel(Diag, inputs);
}
const diag = op({ diag_ });

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
 * Computes the dot product of two matrices and/or vectors, `t1` and `t2`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor2d([[1, 2], [3, 4]]);
 * const c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 *
 * a.dot(b).print();  // or tf.dot(a, b)
 * b.dot(a).print();
 * b.dot(c).print();
 * ```
 * @param t1 The first tensor in the dot operation.
 * @param t2 The second tensor in the dot operation.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function dot_(t1, t2) {
    const $t1 = convertToTensor(t1, 't1', 'dot');
    const $t2 = convertToTensor(t2, 't2', 'dot');
    assert(($t1.rank === 1 || $t1.rank === 2) && ($t2.rank === 1 || $t2.rank === 2), () => `Error in dot: inputs must all be rank 1 or 2, but got ranks ` +
        `${$t1.rank} and ${$t2.rank}.`);
    const t1Inner = ($t1.rank === 1 ? $t1.size : $t1.shape[1]);
    const t2Inner = ($t2.rank === 1 ? $t2.size : $t2.shape[0]);
    assert(t1Inner === t2Inner, () => `Error in dot: inner dimensions of inputs must match, but got ` +
        `${t1Inner} and ${t2Inner}.`);
    if ($t1.rank === 1 && $t2.rank === 1) {
        const t12D = reshape($t1, [1, -1]);
        const t22D = reshape($t2, [-1, 1]);
        const t1t2 = matMul(t12D, t22D);
        return reshape(t1t2, []);
    }
    else if ($t1.rank === 1 && $t2.rank === 2) {
        const t12D = reshape($t1, [1, -1]);
        const t22D = reshape($t2, [$t2.shape[0], $t2.shape[1]]);
        const t1t2 = matMul(t12D, t22D);
        return reshape(t1t2, [t1t2.size]);
    }
    else if ($t1.rank === 2 && $t2.rank === 1) {
        const t22D = reshape($t2, [-1, 1]);
        const t1t2 = matMul($t1, t22D);
        return reshape(t1t2, [t1t2.size]);
    }
    else {
        const t22D = reshape($t2, [$t2.shape[0], $t2.shape[1]]);
        const t1t2 = matMul($t1, t22D);
        return t1t2;
    }
}
const dot = op({ dot_ });

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
 * Returns which elements of x are finite.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isFinite().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function isFinite_(x) {
    const $x = convertToTensor(x, 'x', 'isFinite');
    const inputs = { x: $x };
    return ENGINE.runKernel(IsFinite, inputs);
}
const isFinite$1 = op({ isFinite_ });

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
 * Returns which elements of x are Infinity or -Infinity.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isInf().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function isInf_(x) {
    const $x = convertToTensor(x, 'x', 'isInf');
    const inputs = { x: $x };
    return ENGINE.runKernel(IsInf, inputs);
}
const isInf = op({ isInf_ });

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
 * RReturns which elements of x are NaN.
 *
 * ```js
 * const x = tf.tensor1d([NaN, Infinity, -Infinity, 0, 1]);
 *
 * x.isNaN().print();  // or tf.isNaN(x)
 * ```
 * @param x The input Tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function isNaN_(x) {
    const $x = convertToTensor(x, 'x', 'isNaN');
    const inputs = { x: $x };
    return ENGINE.runKernel(IsNan, inputs);
}
const isNaN$1 = op({ isNaN_ });

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
 * Computes log sigmoid of the input `tf.Tensor` element-wise:
 * `logSigmoid(x)`. For numerical stability, we use `-tf.softplus(-x)`.
 *
 * ```js
 * const x = tf.tensor1d([0, 1, -1, .7]);
 *
 * x.logSigmoid().print();  // or tf.logSigmoid(x)
 * ```
 * @param x The input tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function logSigmoid_(x) {
    const $x = convertToTensor(x, 'x', 'logSigmoid');
    // Use a custom gradient to maintain previous implementation.
    // There is no LogSigmoid kernel in TF so we can't use engine.runKernel
    // directly
    const customOp = customGrad((x) => {
        // TODO(yassogba) we can remove the chained softplus call here only
        // after backends have modualrized softplus at which point we can call
        // engine runKernel(..., Sotfplus, ...) directly.
        const value = neg(softplus(neg(x)));
        const gradFunc = (dy) => {
            const derX = mul(dy, sigmoid(neg(x)));
            return derX;
        };
        return { value, gradFunc };
    });
    return customOp($x);
}
const logSigmoid = op({ logSigmoid_ });

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
 * Returns the truth value of `a XOR b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalXor(b).print();
 * ```
 *
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function logicalXor_(a, b) {
    const $a = convertToTensor(a, 'a', 'logicalXor', 'bool');
    const $b = convertToTensor(b, 'b', 'logicalXor', 'bool');
    assertAndGetBroadcastShape($a.shape, $b.shape);
    // x ^ y = (x | y) & ~(x & y)
    return logicalAnd(logicalOr(a, b), logicalNot(logicalAnd(a, b)));
}
const logicalXor = op({ logicalXor_ });

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
 * Calculates the mean and variance of `x`. The mean and variance are
 * calculated by aggregating the contents of `x` across `axes`. If `x` is
 * 1-D and `axes = [0]` this is just the mean and variance of a vector.
 *
 * @param x The input tensor.
 * @param axis The dimension(s) along with to compute mean and
 *     variance. By default it reduces all dimensions.
 * @param keepDims If true, the moments have the same dimensionality as the
 *     input.
 * @return An object with two keys: `mean` and `variance`.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function moments_(x, axis = null, keepDims = false) {
    x = convertToTensor(x, 'x', 'moments');
    const axes = parseAxisParam(axis, x.shape);
    const xMean = mean(x, axes, keepDims);
    let keepDimsShape = xMean.shape;
    if (!keepDims) {
        keepDimsShape = expandShapeToKeepDim(xMean.shape, axes);
    }
    const devSquared = square(sub(cast(x, 'float32'), reshape(xMean, keepDimsShape)));
    const variance = mean(devSquared, axes, keepDims);
    return { mean: xMean, variance };
}
const moments = op({ moments_ });

/**
 * Computes the next states and outputs of a stack of LSTMCells.
 *
 * Each cell output is used as input to the next cell.
 *
 * Returns `[cellState, cellOutput]`.
 *
 * Derived from tf.contrib.rn.MultiRNNCell.
 *
 * @param lstmCells Array of LSTMCell functions.
 * @param data The input to the cell.
 * @param c Array of previous cell states.
 * @param h Array of previous cell outputs.
 *
 * @doc {heading: 'Operations', subheading: 'RNN'}
 */
function multiRNNCell_(lstmCells, data, c, h) {
    const $data = convertToTensor(data, 'data', 'multiRNNCell');
    const $c = convertToTensorArray(c, 'c', 'multiRNNCell');
    const $h = convertToTensorArray(h, 'h', 'multiRNNCell');
    let input = $data;
    const newStates = [];
    for (let i = 0; i < lstmCells.length; i++) {
        const output = lstmCells[i](input, $c[i], $h[i]);
        newStates.push(output[0]);
        newStates.push(output[1]);
        input = output[1];
    }
    const newC = [];
    const newH = [];
    for (let i = 0; i < newStates.length; i += 2) {
        newC.push(newStates[i]);
        newH.push(newStates[i + 1]);
    }
    return [newC, newH];
}
const multiRNNCell = op({ multiRNNCell_ });

/**
 * Computes the outer product of two vectors, `v1` and `v2`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([3, 4, 5]);
 *
 * tf.outerProduct(a, b).print();
 * ```
 * @param v1 The first vector in the outer product operation.
 * @param v2 The second vector in the outer product operation.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function outerProduct_(v1, v2) {
    const $v1 = convertToTensor(v1, 'v1', 'outerProduct');
    const $v2 = convertToTensor(v2, 'v2', 'outerProduct');
    assert($v1.rank === 1 && $v2.rank === 1, () => `Error in outerProduct: inputs must be rank 1, but got ranks ` +
        `${$v1.rank} and ${$v2.rank}.`);
    const v12D = reshape($v1, [-1, 1]);
    const v22D = reshape($v2, [1, -1]);
    return matMul(v12D, v22D);
}
const outerProduct = op({ outerProduct_ });

/**
 * Pads a `tf.Tensor1D` with a given value and paddings. See `pad` for details.
 */
function pad1d_(x, paddings, constantValue = 0) {
    assert(paddings.length === 2, () => 'Invalid number of paddings. Must be length of 2.');
    return pad(x, [paddings], constantValue);
}
const pad1d = op({ pad1d_ });

/**
 * Pads a `tf.Tensor2D` with a given value and paddings. See `pad` for details.
 */
function pad2d_(x, paddings, constantValue = 0) {
    assert(paddings.length === 2 && paddings[0].length === 2 &&
        paddings[1].length === 2, () => 'Invalid number of paddings. Must be length of 2 each.');
    return pad(x, paddings, constantValue);
}
const pad2d = op({ pad2d_ });

/**
 * Pads a `tf.Tensor3D` with a given value and paddings. See `pad` for details.
 */
function pad3d_(x, paddings, constantValue = 0) {
    assert(paddings.length === 3 && paddings[0].length === 2 &&
        paddings[1].length === 2 && paddings[2].length === 2, () => 'Invalid number of paddings. Must be length of 2 each.');
    return pad(x, paddings, constantValue);
}
const pad3d = op({ pad3d_ });

/**
 * Pads a `tf.Tensor4D` with a given value and paddings. See `pad` for details.
 */
function pad4d_(x, paddings, constantValue = 0) {
    assert(paddings.length === 4 && paddings[0].length === 2 &&
        paddings[1].length === 2 && paddings[2].length === 2 &&
        paddings[3].length === 2, () => 'Invalid number of paddings. Must be length of 2 each.');
    return pad(x, paddings, constantValue);
}
const pad4d = op({ pad4d_ });

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
 * Performs an N-D pooling operation
 *
 * @param input The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param windowShape The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param poolingType The type of pooling, either 'max' or 'avg'.
 * @param pad The type of padding algorithm:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in dilated pooling. Defaults to `[1, 1]`. If `dilationRate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function pool_(input, windowShape, poolingType, pad, dilations, strides) {
    if (dilations == null) {
        dilations = [1, 1];
    }
    if (strides == null) {
        strides = 1;
    }
    if (pad === 0) {
        pad = 'valid';
    }
    const $x = convertToTensor(input, 'x', 'maxPool');
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    assert(eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in pool: Either strides or dilations must be 1. ' +
        `Got strides ${strides} and dilations '${dilations}'`);
    const convInfo = computePool2DInfo(x4D.shape, windowShape, strides, dilations, pad);
    const dilation = [convInfo.dilationHeight, convInfo.dilationWidth];
    // The following implementation does batchToSpace(pool(spaceToBatch(x)))
    // whenever dilation > 1 since the TF kernels do not support dilation > 1.
    // tslint:disable-next-line:max-line-length
    // https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L1037
    let basePadding;
    if (pad === 'same') {
        basePadding = withSpaceToBatchBasePaddings([convInfo.filterHeight, convInfo.filterWidth], dilation);
    }
    else {
        basePadding = [[0, 0], [0, 0]];
    }
    const isDilationOne = dilation[0] === 1 && dilation[1] === 1;
    const [adjustedPadding, adjustedCrops] = requiredSpaceToBatchPaddings([convInfo.inHeight, convInfo.inWidth], dilation, basePadding);
    const convertedPad = isDilationOne ? pad : 'valid';
    const convertedX = isDilationOne ? x4D : spaceToBatchND(x4D, dilation, adjustedPadding);
    const forwardOp = poolingType === 'avg' ?
        () => avgPool(convertedX, windowShape, strides, convertedPad) :
        () => maxPool(convertedX, windowShape, strides, convertedPad);
    const y = forwardOp();
    const res = isDilationOne ? y : batchToSpaceND(y, dilation, adjustedCrops);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
// Helper function to compute crops and paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/array_ops.py#L2184
function requiredSpaceToBatchPaddings(inputShape, blockShape, basePadding) {
    const padStart = basePadding.map(b => b[0]);
    const origPadEnd = basePadding.map(b => b[1]);
    const fullInputShape = inputShape.concat(padStart, origPadEnd);
    const padEndExtra = blockShape.map((b, i) => (b - fullInputShape[i] % b) % b);
    const padEnd = origPadEnd.map((s, i) => s + padEndExtra[i]);
    const paddings = blockShape.map((_, i) => [padStart[i], padEnd[i]]);
    const crops = blockShape.map((_, i) => [0, padEndExtra[i]]);
    return [paddings, crops];
}
// Helper function to compute base paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L524
function withSpaceToBatchBasePaddings(filterShape, dilation) {
    // Spatial dimensions of the filters and the upsampled filters in which we
    // introduce (rate - 1) zeros between consecutive filter values.
    const dilatedFilterShape = filterShape.map((s, i) => {
        return s + (s - 1) * (dilation[i] - 1);
    });
    const padExtraShape = dilatedFilterShape.map(s => s - 1);
    // When padding is odd, we pad more at end, following the same
    // convention as conv2d.
    const padExtraStart = padExtraShape.map(s => Math.floor(s / 2));
    const padExtraEnd = padExtraShape.map((s, i) => s - padExtraStart[i]);
    return padExtraShape.map((_, i) => {
        return [padExtraStart[i], padExtraEnd[i]];
    });
}
const pool = op({ pool_ });

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
 * Creates a `tf.Tensor` with values sampled from a random number generator
 * function defined by the user.
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param randFunction A random number generator function which is called
 * for each element in the output tensor.
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 */
function rand_(shape, randFunction, dtype) {
    const size = sizeFromShape(shape);
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
    for (let i = 0; i < size; i++) {
        values[i] = randFunction();
    }
    return ENGINE.makeTensor(values, shape, dtype);
}
const rand = op({ rand_ });

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
 * Creates a `tf.Tensor` with values sampled from a gamma distribution.
 *
 * ```js
 * tf.randomGamma([2, 2], 1).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param alpha The shape parameter of the gamma distribution.
 * @param beta The inverse scale parameter of the gamma distribution. Defaults
 *     to 1.
 * @param dtype The data type of the output. Defaults to float32.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomGamma_(shape, alpha, beta = 1, dtype = 'float32', seed) {
    if (beta == null) {
        beta = 1;
    }
    if (dtype == null) {
        dtype = 'float32';
    }
    if (dtype !== 'float32' && dtype !== 'int32') {
        throw new Error(`Unsupported data type ${dtype}`);
    }
    const rgamma = new RandGamma(alpha, beta, dtype, seed);
    const res = buffer(shape, dtype);
    for (let i = 0; i < res.values.length; i++) {
        res.values[i] = rgamma.nextValue();
    }
    return res.toTensor();
}
const randomGamma = op({ randomGamma_ });

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
 * Creates a `tf.Tensor` with values sampled from a normal distribution.
 *
 * ```js
 * tf.randomNormal([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param mean The mean of the normal distribution.
 * @param stdDev The standard deviation of the normal distribution.
 * @param dtype The data type of the output.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomNormal_(shape, mean = 0, stdDev = 1, dtype, seed) {
    if (dtype != null && dtype === 'bool') {
        throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss = new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    const res = buffer(shape, dtype);
    for (let i = 0; i < res.values.length; i++) {
        res.values[i] = randGauss.nextValue();
    }
    return res.toTensor();
}
const randomNormal = op({ randomNormal_ });

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
 * Reverses a `tf.Tensor1D`.
 *
 * @param x The input tensor.
 */
function reverse1d_(x) {
    const $x = convertToTensor(x, 'x', 'reverse');
    assert($x.rank === 1, () => `Error in reverse1D: x must be rank 1 but got rank ${$x.rank}.`);
    return reverse($x, 0);
}
const reverse1d = op({ reverse1d_ });

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
 * Reverses a `tf.Tensor2D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse2d_(x, axis) {
    const $x = convertToTensor(x, 'x', 'reverse');
    assert($x.rank === 2, () => `Error in reverse2D: x must be rank 2 but got rank ${$x.rank}.`);
    return reverse($x, axis);
}
const reverse2d = op({ reverse2d_ });

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
 * Reverses a `tf.Tensor3D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse3d_(x, axis) {
    const $x = convertToTensor(x, 'x', 'reverse');
    assert($x.rank === 3, () => `Error in reverse3D: x must be rank 3 but got rank ${$x.rank}.`);
    return reverse($x, axis);
}
const reverse3d = op({ reverse3d_ });

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
 * Reverses a `tf.Tensor4D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse4d_(x, axis) {
    const $x = convertToTensor(x, 'x', 'reverse');
    assert($x.rank === 4, () => `Error in reverse4D: x must be rank 4 but got rank ${$x.rank}.`);
    return reverse($x, axis);
}
const reverse4d = op({ reverse4d_ });

/**
 * 2-D convolution with separable filters.
 *
 * Performs a depthwise convolution that acts separately on channels followed
 * by a pointwise convolution that mixes channels. Note that this is
 * separability between dimensions [1, 2] and 3, not spatial separability
 * between dimensions 1 and 2.
 *
 * See
 * [https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d](
 *     https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)
 * for more details.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param depthwiseFilter The depthwise filter tensor, rank 4, of shape
 *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`. This is
 *     the filter used in the first step.
 * @param pointwiseFilter The pointwise filter tensor, rank 4, of shape
 *     `[1, 1, inChannels * channelMultiplier, outChannels]`. This is
 *     the filter used in the second step.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`. If strides is a single number, then `strideHeight ==
 * strideWidth`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 *
 * @doc {heading: 'Operations', subheading: 'Convolution'}
 */
function separableConv2d_(x, depthwiseFilter, pointwiseFilter, strides, pad, dilation = [1, 1], dataFormat = 'NHWC') {
    const $x = convertToTensor(x, 'x', 'separableConv2d');
    const $depthwiseFilter = convertToTensor(depthwiseFilter, 'depthwiseFilter', 'separableConv2d');
    const $pointwiseFilter = convertToTensor(pointwiseFilter, 'pointwiseFilter', 'separableConv2d');
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    if (dataFormat === 'NCHW') {
        throw new Error('separableConv2d currently does not support dataFormat NCHW; only ' +
            'NHWC is supported');
    }
    assert(x4D.rank === 4, () => `Error in separableConv2d: input must be rank 4, but got ` +
        `rank ${x4D.rank}.`);
    assert($depthwiseFilter.rank === 4, () => `Error in separableConv2d: depthwise filter must be rank 4, but ` +
        `got rank ${$depthwiseFilter.rank}.`);
    assert($pointwiseFilter.rank === 4, () => `Error in separableConv2d: pointwise filter must be rank 4, but ` +
        `got rank ${$depthwiseFilter.rank}.`);
    assert($pointwiseFilter.shape[0] === 1, () => `Error in separableConv2d: the first dimension of pointwise filter ` +
        ` must be 1, but got ${$pointwiseFilter.shape[0]}.`);
    assert($pointwiseFilter.shape[1] === 1, () => `Error in separableConv2d: the second dimension of pointwise ` +
        `filter must be 1, but got ${$pointwiseFilter.shape[1]}.`);
    const inChannels = $depthwiseFilter.shape[2];
    const channelMultiplier = $depthwiseFilter.shape[3];
    assert($pointwiseFilter.shape[2] === inChannels * channelMultiplier, () => `Error in separableConv2d: the third dimension of pointwise filter ` +
        `must be ${inChannels * channelMultiplier}, ` +
        `but got ${$pointwiseFilter.shape[2]}.`);
    const depthwise = depthwiseConv2d(x4D, $depthwiseFilter, strides, pad, dataFormat, dilation);
    const pointwiseStride = 1;
    const res = conv2d(depthwise, $pointwiseFilter, pointwiseStride, 'valid', dataFormat);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
const separableConv2d = op({ separableConv2d_ });

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
 * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
 * of length `size`. See `slice` for details.
 */
function slice1d_(x, begin, size) {
    const $x = convertToTensor(x, 'x', 'slice1d');
    assert($x.rank === 1, () => `slice1d expects a rank-1 tensor, but got a rank-${$x.rank} tensor`);
    return slice($x, [begin], [size]);
}
const slice1d = op({ slice1d_ });

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
 * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice2d_(x, begin, size) {
    const $x = convertToTensor(x, 'x', 'slice2d');
    assert($x.rank === 2, () => `slice2d expects a rank-2 tensor, but got a rank-${$x.rank} tensor`);
    return slice($x, begin, size);
}
const slice2d = op({ slice2d_ });

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
 * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice3d_(x, begin, size) {
    const $x = convertToTensor(x, 'x', 'slice3d');
    assert($x.rank === 3, () => `slice3d expects a rank-3 tensor, but got a rank-${$x.rank} tensor`);
    return slice($x, begin, size);
}
const slice3d = op({ slice3d_ });

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
 * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice4d_(x, begin, size) {
    const $x = convertToTensor(x, 'x', 'slice4d');
    assert($x.rank === 4, () => `slice4d expects a rank-4 tensor, but got a rank-${$x.rank} tensor`);
    return slice($x, begin, size);
}
const slice4d = op({ slice4d_ });

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
 * Creates rank-4 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor4d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor4d([[[[1], [2]], [[3], [4]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor4d(values, shape, dtype) {
    assertNonNull(values);
    if (shape != null && shape.length !== 4) {
        throw new Error('tensor4d() requires shape to have four numbers');
    }
    const inferredShape = inferShape(values, dtype);
    if (inferredShape.length !== 4 && inferredShape.length !== 1) {
        throw new Error('tensor4d() requires values to be number[][][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor4d() requires shape to be provided when `values` ' +
            'are a flat array');
    }
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
/**
 * Creates rank-5 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor5d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor5d([[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor5d(values, shape, dtype) {
    assertNonNull(values);
    if (shape != null && shape.length !== 5) {
        throw new Error('tensor5d() requires shape to have five numbers');
    }
    const inferredShape = inferShape(values, dtype);
    if (inferredShape.length !== 5 && inferredShape.length !== 1) {
        throw new Error('tensor5d() requires values to be ' +
            'number[][][][][] or flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor5d() requires shape to be provided when `values` ' +
            'are a flat array');
    }
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
/**
 * Creates rank-6 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor6d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor6d([[[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function tensor6d(values, shape, dtype) {
    assertNonNull(values);
    if (shape != null && shape.length !== 6) {
        throw new Error('tensor6d() requires shape to have six numbers');
    }
    const inferredShape = inferShape(values, dtype);
    if (inferredShape.length !== 6 && inferredShape.length !== 1) {
        throw new Error('tensor6d() requires values to be number[][][][][][] or ' +
            'flat/TypedArray');
    }
    if (inferredShape.length === 1 && shape == null) {
        throw new Error('tensor6d() requires shape to be provided when `values` ' +
            'are a flat array');
    }
    shape = shape ||
        inferredShape;
    return makeTensor(values, shape, inferredShape, dtype);
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
 * Computes the sum along segments of a `tf.Tensor`.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * const segmentIds = tf.tensor1d([1, 2, 0, 1], 'int32');
 * const numSegments = 3;
 *
 * x.unsortedSegmentSum(segmentIds, numSegments).print()
 * //or tf.unsortedSegmentSum(x, segmentIds, numSegments)
 * ```
 * @param x The `tf.Tensor` that will be summed along its segments.
 * @param segmentIds A `tf.Tensor1D` whose rank is equal to the rank of `x`'s
 * dimension along the `axis`.  Maps each element of `x` to a segment.
 * @param numSegments The number of distinct `segmentIds`.
 *
 * @doc {heading: 'Operations', subheading: 'Segment'}
 */
function unsortedSegmentSum_(x, segmentIds, numSegments) {
    const $x = convertToTensor(x, 'x', 'unsortedSegmentSum');
    const $segmentIds = convertToTensor(segmentIds, 'segmentIds', 'unsortedSegmentSum', 'int32');
    assert(isInt(numSegments), () => 'numSegments must be of dtype int');
    const inputs = { x: $x, segmentIds: $segmentIds };
    const attrs = { numSegments };
    return ENGINE.runKernel(UnsortedSegmentSum, inputs, attrs);
}
const unsortedSegmentSum = op({ unsortedSegmentSum_ });

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
 * Creates a new variable with the provided initial value.
 * ```js
 * const x = tf.variable(tf.tensor([1, 2, 3]));
 * x.assign(tf.tensor([4, 5, 6]));
 *
 * x.print();
 * ```
 *
 * @param initialValue Initial value for the tensor.
 * @param trainable If true, optimizers are allowed to update it.
 * @param name Name of the variable. Defaults to a unique id.
 * @param dtype If set, initialValue will be converted to the given type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function variable(initialValue, trainable = true, name, dtype) {
    return ENGINE.makeVariable(initialValue, trainable, name, dtype);
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
 * Apply boolean mask to tensor.
 *
 * ```js
 * const tensor = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
 * const mask = tf.tensor1d([1, 0, 1], 'bool');
 * const result = await tf.booleanMaskAsync(tensor, mask);
 * result.print();
 * ```
 *
 * @param tensor N-D tensor.
 * @param mask K-D boolean tensor, K <= N and K must be known statically.
 * @param axis A 0-D int Tensor representing the axis in tensor to mask from.
 *     By default, axis is 0 which will mask from the first dimension.
 *     Otherwise K + axis <= N.
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
async function booleanMaskAsync_(tensor, mask, axis) {
    const $tensor = convertToTensor(tensor, 'tensor', 'boolMask');
    const $mask = convertToTensor(mask, 'mask', 'boolMask', 'bool');
    const axisFrom = axis == null ? 0 : axis;
    const maskDim = $mask.rank;
    const tensorShape = $tensor.shape;
    assert(maskDim > 0, () => 'mask cannot be scalar');
    assertShapesMatch(tensorShape.slice(axisFrom, axisFrom + maskDim), $mask.shape, `mask's shape must match the first K dimensions of tensor's shape,`);
    let leadingSize = 1;
    for (let i = axisFrom; i < axisFrom + maskDim; i++) {
        leadingSize *= tensorShape[i];
    }
    const targetTensorShape = tensorShape.slice(0, axisFrom)
        .concat([leadingSize], tensorShape.slice(axisFrom + maskDim));
    const reshapedTensor = reshape($tensor, targetTensorShape);
    const reshapedMask = reshape($mask, [-1]);
    const positivePositions = await whereAsync(reshapedMask);
    const indices = squeeze(positivePositions, [1]);
    const res = gather(reshapedTensor, indices, axisFrom);
    // Ensure no memory leak.
    if (tensor !== $tensor) {
        $tensor.dispose();
    }
    if (mask !== $mask) {
        $mask.dispose();
    }
    indices.dispose();
    reshapedTensor.dispose();
    reshapedMask.dispose();
    positivePositions.dispose();
    return res;
}
const booleanMaskAsync = booleanMaskAsync_;

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
 * Compute the moving average of a variable.
 *
 * Without zeroDebias, the moving average operation is defined by:
 *   `v += delta`
 * where
 *   `delta = (1 - decay) * (x - v)`
 *
 * With zeroDebias (default), the `delta` term is scaled to debias the
 * effect of the (assumed) zero-initialization of `v`.
 *   `delta /= (1 - decay ^ step)`
 *
 * For more details on the zero-debiasing algorithm, see:
 *   https://arxiv.org/abs/1412.6980
 *
 * Note that this function is completely stateless and does not keep track of
 * step count. The step count needs to be maintained by the caller and passed
 * in as `step`.
 *
 * @param v The current moving average value.
 * @param x New input value, must have the same shape and dtype as `v`.
 * @param decay The decay factor. Typical values are 0.95 and 0.99.
 * @param step Step count.
 * @param zeroDebias: Whether zeroDebias is to be performed (default: `true`).
 * @returns The new moving average value.
 *
 * @doc {heading: 'Operations', subheading: 'Moving Average'}
 */
function movingAverage_(v, x, decay, step, zeroDebias = true) {
    const $v = convertToTensor(v, 'v', 'movingAverage');
    const $x = convertToTensor(x, 'x', 'movingAverage');
    const $decay = convertToTensor(decay, 'decay', 'movingAverage');
    assertTypesMatch($v, $x);
    assert(arraysEqual($v.shape, $x.shape), () => 'Shape mismatch in v and x');
    const one = scalar(1);
    const oneMinusDecay = sub(one, $decay);
    let update = mul(sub($x, $v), oneMinusDecay);
    if (zeroDebias) {
        assert(step != null, () => 'When using zeroDebias: true, step is required.');
        const $step = convertToTensor(step, 'step', 'movingAverage');
        update = div(update, sub(one, pow($decay, $step)));
    }
    return add($v, update);
}
const movingAverage = op({ movingAverage_ });

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
 * Normalize noise shape based on provided tensor and noise shape.
 *
 * @param x Tensor.
 * @param noiseShape The shape for the randomly generated keep/drop flags, as
 *   an array of numbers. Optional.
 * @returns Normalized noise shape.
 */
function getNoiseShape(x, noiseShape) {
    if (noiseShape == null) {
        return x.shape.slice();
    }
    if (arraysEqual(x.shape, noiseShape)) {
        return noiseShape;
    }
    if (x.shape.length === noiseShape.length) {
        const newDimension = [];
        for (let i = 0; i < x.shape.length; i++) {
            if (noiseShape[i] == null && x.shape[i] != null) {
                newDimension.push(x.shape[i]);
            }
            else {
                newDimension.push(noiseShape[i]);
            }
        }
        return newDimension;
    }
    return noiseShape;
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
 * Computes dropout.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 2, 1]);
 * const rate = 0.75;
 * const output = tf.dropout(x, rate);
 * output.print();
 * ```
 *
 * @param x A floating point Tensor or TensorLike.
 * @param rate A float in the range [0, 1). The probability that each element
 *   of x is discarded.
 * @param noiseShape An array of numbers of type int32, representing the
 * shape for randomly generated keep/drop flags. If the noiseShape has null
 * value, it will be automatically replaced with the x's relative dimension
 * size. Optional.
 * @param seed Used to create random seeds. Optional.
 * @returns A Tensor of the same shape of x.
 *
 * @doc {heading: 'Operations', subheading: 'Dropout'}
 */
function dropout_(x, rate, noiseShape, seed) {
    const $x = convertToTensor(x, 'x', 'dropout');
    assert($x.dtype === 'float32', () => `x has to be a floating point tensor since it's going to be ` +
        `scaled, but got a ${$x.dtype} tensor instead.`);
    assert(rate >= 0 && rate < 1, () => `rate must be a float in the range [0, 1), but got ${rate}.`);
    if (rate === 0) {
        return x instanceof Tensor ? $x.clone() : $x;
    }
    const $noiseShape = getNoiseShape($x, noiseShape);
    const keepProb = 1 - rate;
    const multiplier = div(floor(add(randomUniform($noiseShape, 0, 1, 'float32', seed), keepProb)), keepProb);
    return mul($x, multiplier);
}
const dropout = op({ dropout_ });

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
 * Returns whether the targets are in the top K predictions.
 *
 * ```js
 * const predictions = tf.tensor2d([[20, 10, 40, 30], [30, 50, -20, 10]]);
 * const targets = tf.tensor1d([2, 0]);
 * const precision = await tf.inTopKAsync(predictions, targets);
 * precision.print();
 * ```
 * @param predictions 2-D or higher `tf.Tensor` with last dimension being
 *     at least `k`.
 * @param targets 1-D or higher `tf.Tensor`.
 * @param k Optional Number of top elements to look at for computing precision,
 *     default to 1.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
async function inTopKAsync_(predictions, targets, k = 1) {
    const $predictions = convertToTensor(predictions, 'predictions', 'inTopK');
    const $targets = convertToTensor(targets, 'targets', 'inTopK');
    assert($predictions.rank > 1, () => 'inTopK() expects the predictions to be of rank 2 or higher, ' +
        `but got ${$predictions.rank}`);
    assert($predictions.rank - 1 === $targets.rank, () => `predictions rank should be 1 larger than ` +
        `targets rank, but got predictions rank ` +
        `${$predictions.rank} and targets rank ${$targets.rank}`);
    assertShapesMatch($predictions.shape.slice(0, $predictions.shape.length - 1), $targets.shape, `predictions's shape should be align with the targets' shape, ` +
        'except the last dimension.');
    const lastDim = $predictions.shape[$predictions.shape.length - 1];
    assert(k > 0 && k <= lastDim, () => `'k' passed to inTopK() must be > 0 && <= the predictions last ` +
        `dimension (${lastDim}), but got ${k}`);
    const predictionsVals = await $predictions.data();
    const targetsVals = await $targets.data();
    // Reshape predictionsVals into a 2d tensor [batch, lastDim]
    // and look up topK along lastDim.
    const [batch, size] = [predictionsVals.length / lastDim, lastDim];
    const precision = getTypedArrayFromDType('bool', batch);
    for (let b = 0; b < batch; b++) {
        const offset = b * size;
        const vals = predictionsVals.subarray(offset, offset + size);
        const valAndInd = [];
        for (let i = 0; i < vals.length; i++) {
            valAndInd.push({ value: vals[i], index: i });
        }
        valAndInd.sort((a, b) => b.value - a.value);
        precision[b] = 0;
        for (let i = 0; i < k; i++) {
            if (valAndInd[i].index === targetsVals[b]) {
                precision[b] = 1;
                break;
            }
        }
    }
    if (predictions !== $predictions) {
        $predictions.dispose();
    }
    if (targets !== $targets) {
        $targets.dispose();
    }
    // Output precision has the same shape as targets.
    return tensor(precision, $targets.shape, 'bool');
}
const inTopKAsync = inTopKAsync_;

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

var fused_ops = /*#__PURE__*/Object.freeze({
    __proto__: null,
    conv2d: conv2d$1,
    depthwiseConv2d: depthwiseConv2d$1,
    matMul: matMul$1
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
/** @doc {heading: 'Training', subheading: 'Classes', namespace: 'train'} */
class Optimizer extends Serializable {
    /**
     * Executes `f()` and minimizes the scalar output of `f()` by computing
     * gradients of y with respect to the list of trainable variables provided by
     * `varList`. If no list is provided, it defaults to all trainable variables.
     *
     * @param f The function to execute and whose output to minimize.
     * @param returnCost Whether to return the scalar cost value produced by
     * executing `f()`.
     * @param varList An optional list of variables to update. If specified, only
     * the trainable variables in varList will be updated by minimize. Defaults to
     * all trainable variables.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers'}
     */
    minimize(f, returnCost = false, varList) {
        const { value, grads } = this.computeGradients(f, varList);
        if (varList != null) {
            const gradArray = varList.map(v => ({ name: v.name, tensor: grads[v.name] }));
            this.applyGradients(gradArray);
        }
        else {
            this.applyGradients(grads);
        }
        // Dispose gradients.
        dispose(grads);
        if (returnCost) {
            return value;
        }
        else {
            value.dispose();
            return null;
        }
    }
    /**
     * The number of iterations that this optimizer instance has been invoked for.
     */
    get iterations() {
        if (this.iterations_ == null) {
            this.iterations_ = 0;
        }
        return this.iterations_;
    }
    incrementIterations() {
        this.iterations_ = this.iterations + 1;
    }
    /**
     * Executes f() and computes the gradient of the scalar output of f() with
     * respect to the list of trainable variables provided by `varList`. If no
     * list is provided, it defaults to all trainable variables.
     *
     * @param f The function to execute and whose output to use for computing
     * gradients with respect to variables.
     * @param varList An optional list of variables to compute gradients with
     * respect to. If specified, only the trainable variables in varList will have
     * gradients computed with respect to. Defaults to all trainable variables.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers'}
     */
    computeGradients(f, varList) {
        return variableGrads(f, varList);
    }
    /**
     * Dispose the variables (if any) owned by this optimizer instance.
     */
    dispose() {
        if (this.iterations_ != null) {
            dispose(this.iterations_);
        }
    }
    async saveIterations() {
        if (this.iterations_ == null) {
            this.iterations_ = 0;
        }
        return {
            name: 'iter',
            // TODO(cais): Use 'int64' type when available.
            tensor: scalar(this.iterations_, 'int32')
        };
    }
    async getWeights() {
        throw new Error('getWeights() is not implemented for this optimizer yet.');
    }
    async setWeights(weightValues) {
        throw new Error(`setWeights() is not implemented for this optimizer class ` +
            `${this.getClassName()}`);
    }
    /**
     * Extract the first element of the weight values and set it
     * as the iterations counter variable of this instance of optimizer.
     *
     * @param weightValues
     * @returns Weight values with the first element consumed and excluded.
     */
    async extractIterations(weightValues) {
        this.iterations_ = (await weightValues[0].tensor.data())[0];
        return weightValues.slice(1);
    }
}
Object.defineProperty(Optimizer, Symbol.hasInstance, {
    value: (instance) => {
        return instance.minimize != null && instance.computeGradients != null &&
            instance.applyGradients != null;
    }
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
/** @doclink Optimizer */
class AdadeltaOptimizer extends Optimizer {
    constructor(learningRate, rho, epsilon = null) {
        super();
        this.learningRate = learningRate;
        this.rho = rho;
        this.epsilon = epsilon;
        this.accumulatedGrads = [];
        this.accumulatedUpdates = [];
        if (epsilon == null) {
            this.epsilon = ENGINE.backend.epsilon();
        }
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        variableNames.forEach((name, i) => {
            const value = ENGINE.registeredVariables[name];
            const trainable = false;
            if (this.accumulatedGrads[i] == null) {
                this.accumulatedGrads[i] = {
                    originalName: `${name}/accum_grad`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            if (this.accumulatedUpdates[i] == null) {
                this.accumulatedUpdates[i] = {
                    originalName: `${name}/accum_var`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            const accumulatedGrad = this.accumulatedGrads[i].variable;
            const accumulatedUpdate = this.accumulatedUpdates[i].variable;
            tidy(() => {
                const newAccumulatedGrad = add(mul(accumulatedGrad, this.rho), mul(square(gradient), 1 - this.rho));
                const updates = mul(div(sqrt(add(accumulatedUpdate, this.epsilon)), sqrt(add(accumulatedGrad, this.epsilon))), gradient);
                const newAccumulatedUpdate = add(mul(accumulatedUpdate, this.rho), mul(square(updates), 1 - this.rho));
                accumulatedGrad.assign(newAccumulatedGrad);
                accumulatedUpdate.assign(newAccumulatedUpdate);
                const newValue = add(mul(updates, -this.learningRate), value);
                value.assign(newValue);
            });
        });
        this.incrementIterations();
    }
    dispose() {
        if (this.accumulatedUpdates != null) {
            dispose(this.accumulatedGrads.map(v => v.variable));
            dispose(this.accumulatedUpdates.map(v => v.variable));
        }
    }
    async getWeights() {
        // Order matters for Python compatibility.
        const variables = [...this.accumulatedGrads, ...this.accumulatedUpdates];
        return [await this.saveIterations()].concat(variables.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        const variableCount = weightValues.length / 2;
        const trainable = false;
        this.accumulatedGrads =
            weightValues.slice(0, variableCount).map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
        this.accumulatedUpdates =
            weightValues.slice(variableCount, variableCount * 2)
                .map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'rho': this.rho,
            'epsilon': this.epsilon
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['rho'], config['epsilon']);
    }
}
/** @nocollapse */
AdadeltaOptimizer.className = 'Adadelta'; // Name matters for Python compatibility.
registerClass(AdadeltaOptimizer);

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
/** @doclink Optimizer */
class AdagradOptimizer extends Optimizer {
    constructor(learningRate, initialAccumulatorValue = 0.1) {
        super();
        this.learningRate = learningRate;
        this.initialAccumulatorValue = initialAccumulatorValue;
        this.accumulatedGrads = [];
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        variableNames.forEach((name, i) => {
            const value = ENGINE.registeredVariables[name];
            if (this.accumulatedGrads[i] == null) {
                const trainable = false;
                this.accumulatedGrads[i] = {
                    originalName: `${name}/accumulator`,
                    variable: tidy(() => fill(value.shape, this.initialAccumulatorValue)
                        .variable(trainable))
                };
            }
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            const accumulatedGrad = this.accumulatedGrads[i].variable;
            tidy(() => {
                const newAccumulatedGrad = add(accumulatedGrad, square(gradient));
                accumulatedGrad.assign(newAccumulatedGrad);
                const newValue = add(mul(div(gradient, sqrt(add(newAccumulatedGrad, ENGINE.backend.epsilon()))), -this.learningRate), value);
                value.assign(newValue);
            });
        });
        this.incrementIterations();
    }
    dispose() {
        if (this.accumulatedGrads != null) {
            dispose(this.accumulatedGrads.map(v => v.variable));
        }
    }
    async getWeights() {
        // Order matters for Python compatibility.
        return [await this.saveIterations()].concat(this.accumulatedGrads.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        const trainable = false;
        this.accumulatedGrads = weightValues.map(v => ({ originalName: v.name, variable: v.tensor.variable(trainable) }));
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'initialAccumulatorValue': this.initialAccumulatorValue,
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['initialAccumulatorValue']);
    }
}
/** @nocollapse */
AdagradOptimizer.className = 'Adagrad'; // Note: Name matters for Python compatibility.
registerClass(AdagradOptimizer);

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
class AdamOptimizer extends Optimizer {
    constructor(learningRate, beta1, beta2, epsilon = null) {
        super();
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.accumulatedFirstMoment = [];
        this.accumulatedSecondMoment = [];
        tidy(() => {
            // accB* will be updated by batch.
            this.accBeta1 = scalar(beta1).variable();
            this.accBeta2 = scalar(beta2).variable();
        });
        if (epsilon == null) {
            this.epsilon = ENGINE.backend.epsilon();
        }
    }
    applyGradients(variableGradients) {
        const varNames = Array.isArray(variableGradients) ?
            variableGradients.map(v => v.name) :
            Object.keys(variableGradients);
        tidy(() => {
            const oneMinusAccBeta1 = sub(1, this.accBeta1);
            const oneMinusAccBeta2 = sub(1, this.accBeta2);
            varNames.forEach((name, i) => {
                const value = ENGINE.registeredVariables[name];
                const trainable = false;
                if (this.accumulatedFirstMoment[i] == null) {
                    this.accumulatedFirstMoment[i] = {
                        originalName: `${name}/m`,
                        variable: tidy(() => zerosLike(value).variable(trainable))
                    };
                }
                if (this.accumulatedSecondMoment[i] == null) {
                    this.accumulatedSecondMoment[i] = {
                        originalName: `${name}/v`,
                        variable: tidy(() => zerosLike(value).variable(trainable))
                    };
                }
                const gradient = Array.isArray(variableGradients) ?
                    variableGradients[i].tensor :
                    variableGradients[name];
                if (gradient == null) {
                    return;
                }
                const firstMoment = this.accumulatedFirstMoment[i].variable;
                const secondMoment = this.accumulatedSecondMoment[i].variable;
                const newFirstMoment = add(mul(firstMoment, this.beta1), mul(gradient, 1 - this.beta1));
                const newSecondMoment = add(mul(secondMoment, this.beta2), mul(square(gradient), 1 - this.beta2));
                const biasCorrectedFirstMoment = div(newFirstMoment, oneMinusAccBeta1);
                const biasCorrectedSecondMoment = div(newSecondMoment, oneMinusAccBeta2);
                firstMoment.assign(newFirstMoment);
                secondMoment.assign(newSecondMoment);
                const newValue = add(mul(div(biasCorrectedFirstMoment, add(sqrt(biasCorrectedSecondMoment), this.epsilon)), -this.learningRate), value);
                value.assign(newValue);
            });
            this.accBeta1.assign(mul(this.accBeta1, this.beta1));
            this.accBeta2.assign(mul(this.accBeta2, this.beta2));
        });
        this.incrementIterations();
    }
    dispose() {
        this.accBeta1.dispose();
        this.accBeta2.dispose();
        if (this.accumulatedFirstMoment != null) {
            dispose(this.accumulatedFirstMoment.map(v => v.variable));
        }
        if (this.accumulatedSecondMoment != null) {
            dispose(this.accumulatedSecondMoment.map(v => v.variable));
        }
    }
    async getWeights() {
        // Order matters for Python compatibility.
        const variables = [...this.accumulatedFirstMoment, ...this.accumulatedSecondMoment];
        return [await this.saveIterations()].concat(variables.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        tidy(() => {
            this.accBeta1.assign(pow(this.beta1, this.iterations_ + 1));
            this.accBeta2.assign(pow(this.beta2, this.iterations_ + 1));
        });
        const variableCount = weightValues.length / 2;
        const trainable = false;
        this.accumulatedFirstMoment =
            weightValues.slice(0, variableCount).map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
        this.accumulatedSecondMoment =
            weightValues.slice(variableCount, variableCount * 2)
                .map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'beta1': this.beta1,
            'beta2': this.beta2,
            'epsilon': this.epsilon,
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['beta1'], config['beta2'], config['epsilon']);
    }
}
/** @nocollapse */
AdamOptimizer.className = 'Adam'; // Note: Name matters for Python compatibility.
registerClass(AdamOptimizer);

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
class AdamaxOptimizer extends Optimizer {
    constructor(learningRate, beta1, beta2, epsilon = null, decay = 0.0) {
        super();
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.decay = decay;
        this.accumulatedFirstMoment = [];
        this.accumulatedWeightedInfNorm = [];
        tidy(() => {
            this.iteration = scalar(0).variable();
            this.accBeta1 = scalar(beta1).variable();
        });
        if (epsilon == null) {
            this.epsilon = ENGINE.backend.epsilon();
        }
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        tidy(() => {
            const oneMinusAccBeta1 = sub(1, this.accBeta1);
            const lr = div(-this.learningRate, add(mul(this.iteration, this.decay), 1));
            variableNames.forEach((name, i) => {
                const value = ENGINE.registeredVariables[name];
                const trainable = false;
                if (this.accumulatedFirstMoment[i] == null) {
                    this.accumulatedFirstMoment[i] = {
                        originalName: `${name}/m`,
                        variable: zerosLike(value).variable(trainable)
                    };
                }
                if (this.accumulatedWeightedInfNorm[i] == null) {
                    this.accumulatedWeightedInfNorm[i] = {
                        originalName: `${name}/v`,
                        variable: zerosLike(value).variable(trainable)
                    };
                }
                const gradient = Array.isArray(variableGradients) ?
                    variableGradients[i].tensor :
                    variableGradients[name];
                if (gradient == null) {
                    return;
                }
                const firstMoment = this.accumulatedFirstMoment[i].variable;
                const weightedInfNorm = this.accumulatedWeightedInfNorm[i].variable;
                const newFirstMoment = add(mul(firstMoment, this.beta1), mul(gradient, 1 - this.beta1));
                const ut0 = mul(weightedInfNorm, this.beta2);
                const ut1 = abs(gradient);
                const newWeightedInfNorm = maximum(ut0, ut1);
                firstMoment.assign(newFirstMoment);
                weightedInfNorm.assign(newWeightedInfNorm);
                const newValue = add(mul(div(lr, oneMinusAccBeta1), div(newFirstMoment, add(newWeightedInfNorm, this.epsilon))), value);
                value.assign(newValue);
            });
            this.iteration.assign(add(this.iteration, 1));
            this.accBeta1.assign(mul(this.accBeta1, this.beta1));
        });
        this.incrementIterations();
    }
    dispose() {
        this.accBeta1.dispose();
        this.iteration.dispose();
        if (this.accumulatedFirstMoment != null) {
            dispose(this.accumulatedFirstMoment.map(v => v.variable));
        }
        if (this.accumulatedWeightedInfNorm != null) {
            dispose(this.accumulatedWeightedInfNorm.map(v => v.variable));
        }
    }
    async getWeights() {
        throw new Error('getWeights() is not implemented for Adamax yet.');
    }
    async setWeights(weightValues) {
        throw new Error('setWeights() is not implemented for Adamax yet.');
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'beta1': this.beta1,
            'beta2': this.beta2,
            'epsilon': this.epsilon,
            'decay': this.decay
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['beta1'], config['beta2'], config['epsilon'], config['decay']);
    }
}
/** @nocollapse */
AdamaxOptimizer.className = 'Adamax'; // Note: Name matters for Python compatbility.
registerClass(AdamaxOptimizer);

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
/** @doclink Optimizer */
class SGDOptimizer extends Optimizer {
    constructor(learningRate) {
        super();
        this.learningRate = learningRate;
        this.setLearningRate(learningRate);
    }
    applyGradients(variableGradients) {
        const varNames = Array.isArray(variableGradients) ?
            variableGradients.map(v => v.name) :
            Object.keys(variableGradients);
        varNames.forEach((name, i) => {
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            const value = ENGINE.registeredVariables[name];
            tidy(() => {
                const newValue = add(mul(this.c, gradient), value);
                value.assign(newValue);
            });
        });
        this.incrementIterations();
    }
    /**
     * Sets the learning rate of the optimizer.
     */
    setLearningRate(learningRate) {
        this.learningRate = learningRate;
        if (this.c != null) {
            this.c.dispose();
        }
        this.c = keep(scalar(-learningRate));
    }
    dispose() {
        this.c.dispose();
    }
    async getWeights() {
        return [await this.saveIterations()];
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        if (weightValues.length !== 0) {
            throw new Error('SGD optimizer does not have settable weights.');
        }
    }
    getConfig() {
        return { 'learningRate': this.learningRate };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate']);
    }
}
/** @nocollapse */
SGDOptimizer.className = 'SGD'; // Note: Name matters for Python compatibility.
registerClass(SGDOptimizer);

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
/** @doclink Optimizer */
class MomentumOptimizer extends SGDOptimizer {
    constructor(learningRate, momentum, useNesterov = false) {
        super(learningRate);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.useNesterov = useNesterov;
        this.accumulations = [];
        this.m = scalar(this.momentum);
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        variableNames.forEach((name, i) => {
            const value = ENGINE.registeredVariables[name];
            if (this.accumulations[i] == null) {
                const trainable = false;
                this.accumulations[i] = {
                    originalName: `${name}/momentum`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            const accumulation = this.accumulations[i].variable;
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            tidy(() => {
                let newValue;
                const newAccumulation = add(mul(this.m, accumulation), gradient);
                if (this.useNesterov) {
                    newValue = add(mul(this.c, add(gradient, mul(newAccumulation, this.m))), value);
                }
                else {
                    newValue = add(mul(this.c, newAccumulation), value);
                }
                accumulation.assign(newAccumulation);
                value.assign(newValue);
            });
        });
        this.incrementIterations();
    }
    dispose() {
        this.m.dispose();
        if (this.accumulations != null) {
            dispose(this.accumulations.map(v => v.variable));
        }
    }
    /**
     * Sets the momentum of the optimizer.
     *
     * @param momentum
     */
    setMomentum(momentum) {
        this.momentum = momentum;
    }
    async getWeights() {
        // Order matters for Python compatibility.
        return [await this.saveIterations()].concat(this.accumulations.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        const trainable = false;
        this.accumulations = weightValues.map(v => ({ originalName: v.name, variable: v.tensor.variable(trainable) }));
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'momentum': this.momentum,
            'useNesterov': this.useNesterov
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['momentum'], config['useNesterov']);
    }
}
/** @nocollapse */
MomentumOptimizer.className = 'Momentum'; // Name matters for Python compatibility.
registerClass(MomentumOptimizer);

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
/** @doclink Optimizer */
class RMSPropOptimizer extends Optimizer {
    constructor(learningRate, decay = 0.9, momentum = 0.0, epsilon = null, centered = false) {
        super();
        this.learningRate = learningRate;
        this.decay = decay;
        this.momentum = momentum;
        this.epsilon = epsilon;
        this.accumulatedMeanSquares = [];
        this.accumulatedMoments = [];
        this.accumulatedMeanGrads = [];
        this.centered = centered;
        if (epsilon == null) {
            this.epsilon = ENGINE.backend.epsilon();
        }
        if (learningRate == null) {
            throw new Error(`learningRate for RMSPropOptimizer must be defined.`);
        }
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        variableNames.forEach((name, i) => {
            const value = ENGINE.registeredVariables[name];
            const trainable = false;
            if (this.accumulatedMeanSquares[i] == null) {
                this.accumulatedMeanSquares[i] = {
                    originalName: `${name}/rms`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            if (this.accumulatedMoments[i] == null) {
                this.accumulatedMoments[i] = {
                    originalName: `${name}/momentum`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            if (this.accumulatedMeanGrads[i] == null && this.centered) {
                this.accumulatedMeanGrads[i] = {
                    originalName: `${name}/mg`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            const accumulatedMeanSquare = this.accumulatedMeanSquares[i].variable;
            const accumulatedMoments = this.accumulatedMoments[i].variable;
            tidy(() => {
                const newAccumulatedMeanSquare = add(mul(accumulatedMeanSquare, this.decay), mul(square(gradient), 1 - this.decay));
                if (this.centered) {
                    const accumulatedMeanGrad = this.accumulatedMeanGrads[i].variable;
                    // Centered gradient
                    const newAccumulatedMeanGrad = add(mul(accumulatedMeanGrad, this.decay), mul(gradient, 1 - this.decay));
                    const gradContribution = div(mul(gradient, this.learningRate), sqrt(sub(newAccumulatedMeanSquare, add(square(newAccumulatedMeanGrad), this.epsilon))));
                    const newAccumulatedMoments = add(mul(accumulatedMoments, this.momentum), gradContribution);
                    accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
                    accumulatedMeanGrad.assign(newAccumulatedMeanGrad);
                    accumulatedMoments.assign(newAccumulatedMoments);
                    const newValue = sub(value, newAccumulatedMoments);
                    value.assign(newValue);
                }
                else {
                    // Plain gradient
                    const newAccumulatedMeanSquare = add(mul(accumulatedMeanSquare, this.decay), mul(square(gradient), 1 - this.decay));
                    const newAccumulatedMoments = add(mul(accumulatedMoments, this.momentum), div(mul(gradient, this.learningRate), sqrt(add(newAccumulatedMeanSquare, this.epsilon))));
                    accumulatedMeanSquare.assign(newAccumulatedMeanSquare);
                    accumulatedMoments.assign(newAccumulatedMoments);
                    const newValue = sub(value, newAccumulatedMoments);
                    value.assign(newValue);
                }
            });
        });
        this.incrementIterations();
    }
    dispose() {
        if (this.accumulatedMeanSquares != null) {
            dispose(this.accumulatedMeanSquares.map(v => v.variable));
        }
        if (this.accumulatedMeanGrads != null && this.centered) {
            dispose(this.accumulatedMeanGrads.map(v => v.variable));
        }
        if (this.accumulatedMoments != null) {
            dispose(this.accumulatedMoments.map(v => v.variable));
        }
    }
    async getWeights() {
        // Order matters for Python compatibility.
        const variables = [...this.accumulatedMeanSquares, ...this.accumulatedMoments];
        if (this.centered) {
            variables.push(...this.accumulatedMeanGrads);
        }
        return [await this.saveIterations()].concat(variables.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        const variableCount = this.centered ? weightValues.length / 3 : weightValues.length / 2;
        const trainable = false;
        this.accumulatedMeanSquares =
            weightValues.slice(0, variableCount).map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
        this.accumulatedMoments =
            weightValues.slice(variableCount, variableCount * 2)
                .map(v => ({
                originalName: v.name,
                variable: v.tensor.variable(trainable)
            }));
        if (this.centered) {
            this.accumulatedMeanGrads =
                weightValues.slice(variableCount * 2, variableCount * 3)
                    .map(v => ({
                    originalName: v.name,
                    variable: v.tensor.variable(trainable)
                }));
        }
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'decay': this.decay,
            'momentum': this.momentum,
            'epsilon': this.epsilon,
            'centered': this.centered
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['decay'], config['momentum'], config['epsilon'], config['centered']);
    }
}
/** @nocollapse */
RMSPropOptimizer.className = 'RMSProp'; // Note: Name matters for Python compatibility.
registerClass(RMSPropOptimizer);

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
class OptimizerConstructors {
    /**
     * Constructs a `tf.SGDOptimizer` that uses stochastic gradient descent.
     *
     * ```js
     * // Fit a quadratic function by learning the coefficients a, b, c.
     * const xs = tf.tensor1d([0, 1, 2, 3]);
     * const ys = tf.tensor1d([1.1, 5.9, 16.8, 33.9]);
     *
     * const a = tf.scalar(Math.random()).variable();
     * const b = tf.scalar(Math.random()).variable();
     * const c = tf.scalar(Math.random()).variable();
     *
     * // y = a * x^2 + b * x + c.
     * const f = x => a.mul(x.square()).add(b.mul(x)).add(c);
     * const loss = (pred, label) => pred.sub(label).square().mean();
     *
     * const learningRate = 0.01;
     * const optimizer = tf.train.sgd(learningRate);
     *
     * // Train the model.
     * for (let i = 0; i < 10; i++) {
     *   optimizer.minimize(() => loss(f(xs), ys));
     * }
     *
     * // Make predictions.
     * console.log(
     *     `a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);
     * const preds = f(xs).dataSync();
     * preds.forEach((pred, i) => {
     *   console.log(`x: ${i}, pred: ${pred}`);
     * });
     * ```
     *
     * @param learningRate The learning rate to use for the SGD algorithm.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static sgd(learningRate) {
        return new SGDOptimizer(learningRate);
    }
    /**
     * Constructs a `tf.MomentumOptimizer` that uses momentum gradient
     * descent.
     *
     * See
     * [http://proceedings.mlr.press/v28/sutskever13.pdf](
     * http://proceedings.mlr.press/v28/sutskever13.pdf)
     *
     * @param learningRate The learning rate to use for the Momentum gradient
     * descent algorithm.
     * @param momentum The momentum to use for the momentum gradient descent
     * algorithm.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static momentum(learningRate, momentum, useNesterov = false) {
        return new MomentumOptimizer(learningRate, momentum, useNesterov);
    }
    /**
     * Constructs a `tf.RMSPropOptimizer` that uses RMSProp gradient
     * descent. This implementation uses plain momentum and is not centered
     * version of RMSProp.
     *
     * See
     * [http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](
     * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
     *
     * @param learningRate The learning rate to use for the RMSProp gradient
     * descent algorithm.
     * @param decay The discounting factor for the history/coming gradient.
     * @param momentum The momentum to use for the RMSProp gradient descent
     * algorithm.
     * @param epsilon Small value to avoid zero denominator.
     * @param centered If true, gradients are normalized by the estimated
     * variance of the gradient.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static rmsprop(learningRate, decay = .9, momentum = 0.0, epsilon = null, centered = false) {
        return new RMSPropOptimizer(learningRate, decay, momentum, epsilon, centered);
    }
    /**
     * Constructs a `tf.AdamOptimizer` that uses the Adam algorithm.
     * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
     *
     * @param learningRate The learning rate to use for the Adam gradient
     * descent algorithm.
     * @param beta1 The exponential decay rate for the 1st moment estimates.
     * @param beta2 The exponential decay rate for the 2nd moment estimates.
     * @param epsilon A small constant for numerical stability.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static adam(learningRate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = null) {
        return new AdamOptimizer(learningRate, beta1, beta2, epsilon);
    }
    /**
     * Constructs a `tf.AdadeltaOptimizer` that uses the Adadelta algorithm.
     * See [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
     *
     * @param learningRate The learning rate to use for the Adadelta gradient
     * descent algorithm.
     * @param rho The learning rate decay over each update.
     * @param epsilon A constant epsilon used to better condition the grad
     * update.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static adadelta(learningRate = .001, rho = .95, epsilon = null) {
        return new AdadeltaOptimizer(learningRate, rho, epsilon);
    }
    /**
     * Constructs a `tf.AdamaxOptimizer` that uses the Adamax algorithm.
     * See [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
     *
     * @param learningRate The learning rate to use for the Adamax gradient
     * descent algorithm.
     * @param beta1 The exponential decay rate for the 1st moment estimates.
     * @param beta2 The exponential decay rate for the 2nd moment estimates.
     * @param epsilon A small constant for numerical stability.
     * @param decay The learning rate decay over each update.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static adamax(learningRate = 0.002, beta1 = 0.9, beta2 = 0.999, epsilon = null, decay = 0.0) {
        return new AdamaxOptimizer(learningRate, beta1, beta2, epsilon, decay);
    }
    /**
     * Constructs a `tf.AdagradOptimizer` that uses the Adagrad algorithm.
     * See
     * [http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](
     * http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
     * or
     * [http://ruder.io/optimizing-gradient-descent/index.html#adagrad](
     * http://ruder.io/optimizing-gradient-descent/index.html#adagrad)
     *
     * @param learningRate The learning rate to use for the Adagrad gradient
     * descent algorithm.
     * @param initialAccumulatorValue Starting value for the accumulators, must be
     * positive.
     *
     * @doc {heading: 'Training', subheading: 'Optimizers', namespace: 'train'}
     */
    static adagrad(learningRate, initialAccumulatorValue = 0.1) {
        return new AdagradOptimizer(learningRate, initialAccumulatorValue);
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
const train = {
    sgd: OptimizerConstructors.sgd,
    momentum: OptimizerConstructors.momentum,
    adadelta: OptimizerConstructors.adadelta,
    adagrad: OptimizerConstructors.adagrad,
    rmsprop: OptimizerConstructors.rmsprop,
    adamax: OptimizerConstructors.adamax,
    adam: OptimizerConstructors.adam
};

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
const delayCallback = (() => {
    if (typeof requestAnimationFrame !== 'undefined') {
        return requestAnimationFrame;
    }
    else if (typeof setImmediate !== 'undefined') {
        return setImmediate;
    }
    return (f) => f(); // no delays
})();
/**
 * Returns a promise that resolve when a requestAnimationFrame has completed.
 *
 * On Node.js this uses setImmediate instead of requestAnimationFrame.
 *
 * This is simply a sugar method so that users can do the following:
 * `await tf.nextFrame();`
 *
 * @doc {heading: 'Performance', subheading: 'Timing'}
 */
function nextFrame() {
    return new Promise(resolve => delayCallback(() => resolve()));
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

var kernel_impls = /*#__PURE__*/Object.freeze({
    __proto__: null,
    nonMaxSuppressionV3Impl: nonMaxSuppressionV3Impl,
    nonMaxSuppressionV4Impl: nonMaxSuppressionV4Impl,
    nonMaxSuppressionV5Impl: nonMaxSuppressionV5Impl,
    whereImpl: whereImpl
});

export { AdadeltaOptimizer, AdagradOptimizer, AdamOptimizer, AdamaxOptimizer, MomentumOptimizer, Optimizer, RMSPropOptimizer, SGDOptimizer, basicLSTMCell, batchNorm2d, batchNorm3d, batchNorm4d, booleanMaskAsync, browser, concat1d, concat2d, concat3d, concat4d, conv3dTranspose, diag, dot, dropout, fused_ops as fused, inTopKAsync, io, isFinite$1 as isFinite, isInf, isNaN$1 as isNaN, kernel_impls, logSigmoid, logicalXor, math, moments, movingAverage, multiRNNCell, nextFrame, outerProduct, pad1d, pad2d, pad3d, pad4d, pool, rand, randomGamma, randomNormal, reverse1d, reverse2d, reverse3d, reverse4d, separableConv2d, serialization, slice1d, slice2d, slice3d, slice4d, tensor3d, tensor4d, tensor5d, tensor6d, test_util, train, unsortedSegmentSum, variable, version as version_core };
