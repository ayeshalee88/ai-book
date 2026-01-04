import { useState, useCallback } from 'react';
import BackendClient from '../services/BackendClient';

interface ApiState {
  loading: boolean;
  error: string | null;
  data: any;
}

const useApi = (baseUrl?: string) => {
  const [state, setState] = useState<ApiState>({
    loading: false,
    error: null,
    data: null,
  });

  const client = new BackendClient(baseUrl);

  const execute = useCallback(async (question: string) => {
    setState({ loading: true, error: null, data: null });

    try {
      const response = await client.query(question);
      setState({ loading: false, error: null, data: response });
      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      setState({ loading: false, error: errorMessage, data: null });
      throw error;
    }
  }, [client]);

  const reset = useCallback(() => {
    setState({ loading: false, error: null, data: null });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
};

export default useApi;