import {
  IPredictionRequest,
  IPredictionResponse,
  IPredictionRequestResponse
} from './types';
import httpClient from '../services/client';

export const predictImage = async (
  image: IPredictionRequest,
): Promise<IPredictionResponse> => {
  return await httpClient.publicInstance
    .post('/predict/', image)
    .then((response: IPredictionRequestResponse) => response.data)
    .catch((error: unknown) => {
      throw error;
    });
};
