interface ApiResponse {
  answer: string;
  sources?: string[];
}

class BackendClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = 'http://127.0.0.1:8000/api/v1/qa', timeout: number = 30000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  async query(question: string): Promise<ApiResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      return data;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Network error: Unable to connect to the server');
      }

      if (error.name === 'AbortError') {
        throw new Error('Request timeout: The request took too long to complete');
      }

      throw error;
    }
  }
}

export default BackendClient;