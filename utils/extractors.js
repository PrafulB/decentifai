export const extractLocalParameters = (backend) => {
    if (backend === 'tfjs') {
        return extractTFJSModelParameters
    }
    else if (backend === 'onnx') {
        return extractONNXModelParameters
    }

    throw new Error(`Unsupported backend: ${backend}`);
}

const extractTFJSModelParameters = (model) => {

    const weights = {};
    const modelWeights = model.getWeights();
    const layers = model.layers || [];

    let weightIndex = 0;

    for (const layer of layers) {
        const layerName = layer.name;
        const layerWeights = [];

        const numWeights = layer.trainableWeights ? layer.trainableWeights.length : 0;

        for (let i = 0; i < numWeights; i++) {
            // Check if we have enough weights
            if (weightIndex < modelWeights.length) {
                const tensor = modelWeights[weightIndex];
                const values = Array.from(tensor.dataSync());
                const shape = tensor.shape;

                layerWeights.push({
                    values,
                    shape
                });

                weightIndex++;
            }
        }

        if (layerWeights.length > 0) {
            weights[layerName] = layerWeights;
        }
    }

    return weights;
}

const extractONNXModelParameters = (model) => {

    const weights = {};

    try {
        // Get all parameter names from the model
        const session = model.session;
        if (!session) {
            throw new Error('ONNX Session not found in model');
        }

        // Get model metadata to find parameter tensor names
        const modelProto = session._model;
        if (!modelProto || !modelProto.graph || !modelProto.graph.initializer) {
            throw new Error('Unable to access ONNX model graph');
        }

        // Extract weights from initializers
        for (const initializer of modelProto.graph.initializer) {
            const name = initializer.name;

            // Skip non-trainable parameters
            if (name.includes('constant') || name.includes('shape')) {
                continue;
            }

            const dataType = initializer.data_type;
            const dimensions = initializer.dims;

            let values;
            if (initializer.raw_data) {
                const buffer = initializer.raw_data.buffer || initializer.raw_data;
                values = Array.from(new Float32Array(buffer));
            } else if (initializer.float_data) {
                values = Array.from(initializer.float_data);
            } else {
                // Skip if values can't be extracted for some reason
                continue;
            }

            weights[name] = {
                values,
                shape: dimensions,
                dataType
            };
        }
    } catch (error) {
        throw new Error(`Failed to extract ONNX parameters: ${error.message}`);
    }

    return weights;
}