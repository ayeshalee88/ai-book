interface ApiResponse {
  answer: string;
  sources?: string[];
}

class BackendClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = 'http://127.0.0.1:8000/api/v1', timeout: number = 30000) {
    console.log('╔══════════════════════════════════════════════╗');
    console.log('║ BACKEND CLIENT INITIALIZED WITH URL:');
    console.log('║ →', baseUrl);
    console.log('╚══════════════════════════════════════════════╝');
    
    console.log('🌐 USING BASE URL:', baseUrl);
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  async query(question: string): Promise<ApiResponse> {
    console.log('╔══════════════════════════════════════════════╗');
    console.log('║ ACTUAL FETCH URL BEING CALLED:');
    console.log('║ →', this.baseUrl);
    console.log('║ METHOD: POST');
    console.log('╚══════════════════════════════════════════════╝');

    console.log('🔍 ATTEMPTING TO QUERY:', this.baseUrl);
    console.log('📝 QUESTION:', question);
    
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
      
      console.log('📡 RESPONSE STATUS:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ ERROR RESPONSE:', errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      console.log('✅ SUCCESS DATA:', data);
      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      console.error('💥 FULL ERROR:', error);

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