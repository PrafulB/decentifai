export const updateLocalParameters = (backend) => {
    if (backend === 'tfjs') {
        return updateLocalTFJSModelParameters
    } else if (backend === 'onnx') {
        return updateLocalONNXModelParameters
    }

    throw new Error(`Unsupported backend: ${backend}`);
}

const updateLocalTFJSModelParameters = (model, parameters) => {
            const layers = model.layers || [];
            const newWeights = [];

            for (const layer of layers) {
                const layerName = layer.name;
                const layerParams = parameters[layerName];

                if (!layerParams || !layerParams.length) {
                    continue;
                }

                // Convert parameters back to tensors
                for (const param of layerParams) {
                    const { values, shape } = param;

                    const tensor = tf.tensor(values, shape);
                    newWeights.push(tensor);
                }
            }

            // Update model weights if we have any
            if (newWeights.length > 0) {
                model.setWeights(newWeights);
            }

            return true;
}

const updateLocalONNXModelParameters = (model, parameters) => {
    // TODO
}