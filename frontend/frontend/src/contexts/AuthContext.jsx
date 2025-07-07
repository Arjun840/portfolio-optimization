import React, { createContext, useContext, useState, useEffect } from 'react';
import authService from '../services/authService';
import toast from 'react-hot-toast';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Check authentication status on app load
  useEffect(() => {
    const token = authService.getToken();
    if (token) {
      setIsAuthenticated(true);
      setUser({ token }); // In a real app, you might decode the JWT to get user info
    }
    setIsLoading(false);
  }, []);

  const login = async (email, password) => {
    try {
      setIsLoading(true);
      const result = await authService.login(email, password);
      
      if (result.success) {
        setIsAuthenticated(true);
        setUser({ 
          email,
          token: result.data.token,
          expiresIn: result.data.expiresIn 
        });
        toast.success('Successfully logged in!');
        return { success: true };
      } else {
        toast.error(result.error);
        return { success: false, error: result.error };
      }
    } catch (error) {
      const errorMessage = 'Login failed. Please try again.';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setIsLoading(false);
    }
  };

  const signup = async (email, password, fullName) => {
    try {
      setIsLoading(true);
      const result = await authService.signup(email, password, fullName);
      
      if (result.success) {
        setIsAuthenticated(true);
        setUser({ 
          email,
          fullName,
          token: result.data.token,
          expiresIn: result.data.expiresIn 
        });
        toast.success('Account created successfully!');
        return { success: true };
      } else {
        toast.error(result.error);
        return { success: false, error: result.error };
      }
    } catch (error) {
      const errorMessage = 'Signup failed. Please try again.';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    authService.logout();
    setIsAuthenticated(false);
    setUser(null);
    toast.success('Successfully logged out');
  };

  const value = {
    user,
    isAuthenticated,
    isLoading,
    login,
    signup,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}; 