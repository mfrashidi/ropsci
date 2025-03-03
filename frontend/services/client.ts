import axios from 'axios';
export const baseUrl = 'localhost';

const defaultOptions = {
  baseURL: `http://${baseUrl}:8001`,
  headers: {
    'Content-Type': 'application/json',
  },
};
const httpClient = () => {
  const publicInstance = axios.create(defaultOptions);
  publicInstance.interceptors.response.use(
    function (response) {
      return response;
    },
    function (error) {
      return Promise.reject(error);
    },
  );
  return { publicInstance };
};

export default httpClient();
