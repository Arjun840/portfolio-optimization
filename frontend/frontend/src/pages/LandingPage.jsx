import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { 
  TrendingUp, 
  BarChart3, 
  PieChart, 
  Target, 
  Shield,
  Zap,
  Brain,
  DollarSign,
  ArrowRight,
  CheckCircle,
  Users,
  Award
} from 'lucide-react';

const LandingPage = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: "AI-Enhanced Returns",
      description: "Advanced machine learning algorithms predict optimal asset allocation with sophisticated risk modeling."
    },
    {
      icon: Shield,
      title: "Risk Management",
      description: "Multi-layered constraint systems ensure your portfolio stays within your risk tolerance and investment goals."
    },
    {
      icon: Target,
      title: "Custom Optimization",
      description: "Build your own portfolio and let our algorithms optimize it, or start fresh with AI-generated allocations."
    },
    {
      icon: BarChart3,
      title: "Real-Time Analytics",
      description: "Live performance tracking, efficient frontier analysis, and comprehensive portfolio metrics dashboard."
    }
  ];

  const stats = [
    { number: "2.69", label: "Avg Sharpe Ratio", suffix: "" },
    { number: "40.2", label: "Expected Return", suffix: "%" },
    { number: "45", label: "Assets Analyzed", suffix: "+" },
    { number: "14.2", label: "Risk Level", suffix: "%" }
  ];

  const testimonials = [
    {
      quote: "PortfolioMax transformed how I think about portfolio optimization. The AI insights are incredible.",
      author: "Sarah Chen",
      role: "Quantitative Analyst"
    },
    {
      quote: "Finally, a tool that combines academic rigor with practical usability. The custom portfolio builder is game-changing.",
      author: "Michael Rodriguez",
      role: "Portfolio Manager"
    },
    {
      quote: "The risk management features give me confidence in every allocation decision.",
      author: "Jennifer Kim",
      role: "Financial Advisor"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="h-10 w-10 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center mr-3">
                <div className="relative">
                  <TrendingUp className="h-6 w-6 text-white" />
                  <div className="absolute -top-1 -right-1 h-3 w-3 bg-green-400 rounded-full animate-pulse"></div>
                </div>
              </div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-blue-700 bg-clip-text text-transparent">
                PortfolioMax
              </h1>
            </div>
            
            <nav className="hidden md:flex space-x-8">
              <a href="#features" className="text-gray-600 hover:text-blue-600 transition-colors">Features</a>
              <a href="#how-it-works" className="text-gray-600 hover:text-blue-600 transition-colors">How it Works</a>
              <a href="#testimonials" className="text-gray-600 hover:text-blue-600 transition-colors">Testimonials</a>
            </nav>

            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/login')}
                className="text-gray-600 hover:text-blue-600 transition-colors"
              >
                Sign In
              </button>
              <button
                onClick={() => navigate('/signup')}
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Get Started
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-20 pb-32 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h2 className="text-5xl md:text-7xl font-bold mb-8">
              <span className="block text-gray-900">Optimize Everything.</span>
              <span className="block bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">
                Before You Invest.
              </span>
            </h2>
            
            <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed">
              PortfolioMax is an AI-powered portfolio optimization platform that sees your investments, 
              understands your risk tolerance, and delivers optimal allocations — in real time.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center mb-16">
              <motion.button
                onClick={() => navigate('/dashboard')}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors flex items-center justify-center"
              >
                Start Optimizing
                <ArrowRight className="ml-2 h-5 w-5" />
              </motion.button>
              <motion.button
                onClick={() => navigate('/signup')}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="border-2 border-blue-600 text-blue-600 px-8 py-4 rounded-lg text-lg font-semibold hover:bg-blue-50 transition-colors"
              >
                Try Demo Account
              </motion.button>
            </div>

            {/* Demo Account Info */}
            <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-6 max-w-md mx-auto border border-blue-200">
              <p className="text-sm text-gray-600 mb-3">Demo Account Credentials:</p>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Email:</span>
                  <span className="font-mono text-blue-600">demo@portfoliomax.com</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Password:</span>
                  <span className="font-mono text-blue-600">demo123</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h3 className="text-3xl font-bold text-gray-900 mb-4">
              Proven Performance
            </h3>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Our AI-driven optimization delivers consistently superior results across all market conditions.
            </p>
          </motion.div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                className="text-center"
              >
                <div className="text-4xl font-bold text-blue-600 mb-2">
                  {stat.number}{stat.suffix}
                </div>
                <div className="text-gray-600">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h3 className="text-4xl font-bold text-gray-900 mb-4">
              Everything You Need. Before You Invest.
            </h3>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              PortfolioMax combines cutting-edge AI with institutional-grade portfolio theory to optimize your investments.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1, duration: 0.6 }}
                  className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 border border-gray-200 hover:border-blue-300 transition-all duration-300 hover:shadow-xl"
                >
                  <div className="flex items-center mb-6">
                    <div className="h-12 w-12 bg-blue-100 rounded-lg flex items-center justify-center mr-4">
                      <Icon className="h-6 w-6 text-blue-600" />
                    </div>
                    <h4 className="text-xl font-semibold text-gray-900">{feature.title}</h4>
                  </div>
                  <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h3 className="text-4xl font-bold text-gray-900 mb-4">
              Three Ways PortfolioMax Changes How You Invest
            </h3>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.6 }}
              className="text-center"
            >
              <div className="h-16 w-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Zap className="h-8 w-8 text-blue-600" />
              </div>
              <h4 className="text-xl font-semibold text-gray-900 mb-4">Instant Optimization</h4>
              <p className="text-gray-600">
                Enter your risk tolerance and investment goals. Our AI instantly generates optimal portfolio allocations with real-time market data.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
              className="text-center"
            >
              <div className="h-16 w-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Target className="h-8 w-8 text-blue-600" />
              </div>
              <h4 className="text-xl font-semibold text-gray-900 mb-4">Custom Portfolio Builder</h4>
              <p className="text-gray-600">
                Start with your existing holdings or build from scratch. Our system optimizes your selections while respecting your preferences.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              className="text-center"
            >
              <div className="h-16 w-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <PieChart className="h-8 w-8 text-blue-600" />
              </div>
              <h4 className="text-xl font-semibold text-gray-900 mb-4">Advanced Analytics</h4>
              <p className="text-gray-600">
                Visualize efficient frontiers, track performance metrics, and analyze risk-return profiles with institutional-grade tools.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h3 className="text-4xl font-bold text-gray-900 mb-4">
              "This feels like having a quantitative analyst on demand."
            </h3>
            <p className="text-xl text-gray-600">We agree.</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.6 }}
                className="bg-white/80 backdrop-blur-sm rounded-2xl p-8 border border-gray-200"
              >
                <p className="text-gray-700 mb-6 leading-relaxed">"{testimonial.quote}"</p>
                <div>
                  <div className="font-semibold text-gray-900">{testimonial.author}</div>
                  <div className="text-gray-500">{testimonial.role}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 to-blue-700">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h3 className="text-4xl font-bold text-white mb-4">
              Welcome to
            </h3>
            <h2 className="text-6xl font-bold text-white mb-8">
              The Future of Portfolio Optimization.
            </h2>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <motion.button
                onClick={() => navigate('/dashboard')}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="bg-white text-blue-600 px-8 py-4 rounded-lg text-lg font-semibold hover:bg-gray-50 transition-colors"
              >
                Start Optimizing Now
              </motion.button>
              <motion.button
                onClick={() => navigate('/signup')}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="border-2 border-white text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-white/10 transition-colors"
              >
                Create Account
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center mb-4">
                <div className="h-8 w-8 bg-blue-600 rounded-lg flex items-center justify-center mr-3">
                  <TrendingUp className="h-5 w-5 text-white" />
                </div>
                <span className="text-xl font-bold">PortfolioMax</span>
              </div>
              <p className="text-gray-400">
                AI-powered portfolio optimization for the modern investor.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#features" className="hover:text-white transition-colors">Features</a></li>
                <li><a href="#how-it-works" className="hover:text-white transition-colors">How it Works</a></li>
                <li><a href="#" className="hover:text-white transition-colors">API</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Security</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">About</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Legal</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Terms of Service</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Cookie Policy</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2025 PortfolioMax. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage; 