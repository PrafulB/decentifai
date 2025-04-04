export const federatedAveraging = (peerParameters, backend) => {

    if (!peerParameters || peerParameters.length === 0) {
      throw new Error('No parameters to aggregate');
    }

    const aggregated = {};
    const allParams = peerParameters.map(peer => peer.parameters);

    const keys = Object.keys(allParams[0]);

    for (const key of keys) {
      // If the parameter is an array of layer weights (TensorFlow.js style)
      if (backend === 'tfjs') {
        aggregated[key] = [];

        for (let i = 0; i < allParams[0][key].length; i++) {
          const tensorParams = allParams[0][key][i];
          const shape = tensorParams.shape;
          const valueLength = tensorParams.values.length;

          const aggregatedValues = new Array(valueLength).fill(0);

          // Sum all peer values
          for (const params of allParams) {
            if (params[key] && params[key][i] && params[key][i].values) {
              const values = params[key][i].values;
              for (let j = 0; j < valueLength; j++) {
                aggregatedValues[j] += values[j];
              }
            }
          }

          // Average the values
          for (let j = 0; j < aggregatedValues.length; j++) {
            aggregatedValues[j] /= allParams.length;
          }

          aggregated[key].push({
            values: aggregatedValues,
            shape: shape
          });
        }
      }

      else if (backend === 'onnx') {
        // TEST THIS!!!
        const firstParam = allParams[0][key];
        const shape = firstParam.shape;
        const dataType = firstParam.dataType;
        const valueLength = firstParam.values.length;

        const aggregatedValues = new Array(valueLength).fill(0);

        // Sum all peer values
        for (const params of allParams) {
          if (params[key] && params[key].values) {
            const values = params[key].values;
            for (let i = 0; i < valueLength; i++) {
              aggregatedValues[i] += values[i];
            }
          }
        }

        // Average the values
        for (let i = 0; i < aggregatedValues.length; i++) {
          aggregatedValues[i] /= allParams.length;
        }

        aggregated[key] = {
          values: aggregatedValues,
          shape: shape,
          dataType: dataType
        };
      }
    }

    return aggregated;
  };