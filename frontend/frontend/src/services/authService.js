import axios from 'axios';
import Cookies from 'js-cookie';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class AuthService {
  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor to include auth token
    this.api.interceptors.request.use(
      (config) => {
        const token = this.getToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Add response interceptor to handle token expiration
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          this.logout();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Token Management
  setToken(token) {
    Cookies.set('auth_token', token, { 
      expires: 0.5, // 0.5 days = 12 hours
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict'
    });
  }

  getToken() {
    return Cookies.get('auth_token');
  }

  removeToken() {
    Cookies.remove('auth_token');
  }

  isAuthenticated() {
    return !!this.getToken();
  }

  // Authentication Methods
  async login(email, password) {
    try {
      const response = await this.api.post('/auth/login', {
        email,
        password,
      });
      
      const { access_token, token_type, expires_in } = response.data;
      this.setToken(access_token);
      
      return {
        success: true,
        data: {
          token: access_token,
          type: token_type,
          expiresIn: expires_in,
        },
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Login failed',
      };
    }
  }

  async signup(email, password, fullName) {
    try {
      const response = await this.api.post('/auth/signup', {
        email,
        password,
        full_name: fullName,
      });
      
      const { access_token, token_type, expires_in } = response.data;
      this.setToken(access_token);
      
      return {
        success: true,
        data: {
          token: access_token,
          type: token_type,
          expiresIn: expires_in,
        },
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Signup failed',
      };
    }
  }

  logout() {
    this.removeToken();
  }

  // Portfolio API Methods
  async optimizePortfolio(optimizationParams) {
    try {
      const response = await this.api.post('/optimize', optimizationParams);
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Portfolio optimization failed',
      };
    }
  }

  async optimizeCustomPortfolio(request, portfolioValue = 100000) {
    try {
      const response = await this.api.post(`/optimize/custom?portfolio_value=${portfolioValue}`, request);
      return response.data;
    } catch (error) {
      throw error;
    }
  }

  async getStocks() {
    try {
      const response = await this.api.get('/data/stocks');
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to fetch stocks',
      };
    }
  }

  async getClusters() {
    try {
      const response = await this.api.get('/data/clusters');
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to fetch clusters',
      };
    }
  }

  async getEfficientFrontier(constraintLevel = 'enhanced', returnMethod = 'ml_enhanced') {
    try {
      const response = await this.api.get('/data/efficient-frontier', {
        params: {
          constraint_level: constraintLevel,
          return_method: returnMethod,
        },
      });
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to fetch efficient frontier',
      };
    }
  }

  async getHistoricalPerformance(portfolioId, benchmark = 'SPY') {
    try {
      const response = await this.api.get(`/data/historical/${portfolioId}`, {
        params: { benchmark },
      });
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to fetch historical performance',
      };
    }
  }

  // Health Check
  async healthCheck() {
    try {
      const response = await this.api.get('/');
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Health check failed',
      };
    }
  }

  // Portfolio Management Methods
  async savePortfolio(portfolioName, description, portfolioData) {
    try {
      const response = await this.api.post('/portfolios/save', {
        portfolio_name: portfolioName,
        description: description,
        portfolio_data: portfolioData,
      });
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to save portfolio',
      };
    }
  }

  async getSavedPortfolios() {
    try {
      const response = await this.api.get('/portfolios');
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to fetch saved portfolios',
      };
    }
  }

  async getSavedPortfolio(savedId) {
    try {
      const response = await this.api.get(`/portfolios/${savedId}`);
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to fetch saved portfolio',
      };
    }
  }

  async updateSavedPortfolio(savedId, portfolioName, description) {
    try {
      const response = await this.api.put(`/portfolios/${savedId}`, {
        portfolio_name: portfolioName,
        description: description,
      });
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to update portfolio',
      };
    }
  }

  async deleteSavedPortfolio(savedId) {
    try {
      const response = await this.api.delete(`/portfolios/${savedId}`);
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to delete portfolio',
      };
    }
  }

  async comparePortfolios(portfolioIds) {
    try {
      const response = await this.api.post('/portfolios/compare', {
        portfolio_ids: portfolioIds,
      });
      return {
        success: true,
        data: response.data,
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to compare portfolios',
      };
    }
  }
}

// Create singleton instance
const authService = new AuthService();
export default authService; 