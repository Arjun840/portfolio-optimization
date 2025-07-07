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


// Technology Icons
const PythonIcon = ({ className = "h-8 w-8" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none">
    <path d="M12 2C10.2 2 8.8 2.8 8.8 4.4V7.2H12.8V7.6H6.4C4.8 7.6 3.6 8.8 3.6 10.4V13.6C3.6 15.2 4.8 16.4 6.4 16.4H8V13.6C8 12 9.2 10.8 10.8 10.8H15.2C16.8 10.8 18 9.6 18 8V4.4C18 2.8 16.8 1.6 15.2 1.6H12V2Z" fill="#3776ab"/>
    <path d="M12 22C13.8 22 15.2 21.2 15.2 19.6V16.8H11.2V16.4H17.6C19.2 16.4 20.4 15.2 20.4 13.6V10.4C20.4 8.8 19.2 7.6 17.6 7.6H16V10.4C16 12 14.8 13.2 13.2 13.2H8.8C7.2 13.2 6 14.4 6 16V19.6C6 21.2 7.2 22.4 8.8 22.4H12V22Z" fill="#ffd343"/>
    <circle cx="10" cy="5" r="1" fill="#ffd343"/>
    <circle cx="14" cy="19" r="1" fill="#3776ab"/>
  </svg>
);

const ReactIcon = ({ className = "h-8 w-8" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="#61dafb">
    <circle cx="12" cy="12" r="2" />
    <path d="M12 2C8 2 4 4 4 7.5C4 11 8 13 12 13C16 13 20 11 20 7.5C20 4 16 2 12 2Z" stroke="#61dafb" strokeWidth="1" fill="none"/>
    <path d="M12 11C16 11 20 13 20 16.5C20 20 16 22 12 22C8 22 4 20 4 16.5C4 13 8 11 12 11Z" stroke="#61dafb" strokeWidth="1" fill="none"/>
    <path d="M5 9.5C5 6 7.5 2.5 12 2.5C16.5 2.5 19 6 19 9.5C19 13 16.5 16.5 12 16.5C7.5 16.5 5 13 5 9.5Z" stroke="#61dafb" strokeWidth="1" fill="none"/>
  </svg>
);

const FastAPIIcon = ({ className = "h-8 w-8" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="#009688">
    <path d="M12 2L22 12L12 22L2 12L12 2Z" fill="#009688"/>
    <path d="M8 12L12 8L16 12L12 16L8 12Z" fill="white"/>
  </svg>
);

const PostgreSQLIcon = ({ className = "h-8 w-8" }) => (
  <svg className={className} viewBox="0 0 24 24" fill="#336791">
    <path d="M17.128 0c-.324 0-.648.125-.896.373L12 4.606 7.768.373C7.52.125 7.196 0 6.872 0 6.22 0 5.696.524 5.696 1.176v21.648C5.696 23.476 6.22 24 6.872 24c.324 0 .648-.125.896-.373L12 19.394l4.232 4.233c.248.248.572.373.896.373.652 0 1.176-.524 1.176-1.176V1.176C18.304.524 17.78 0 17.128 0z"/>
  </svg>
);

const LandingPage = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: "AI-Enhanced Returns",
      description: "Advanced machine learning algorithms predict optimal asset allocation with sophisticated risk modeling.",
      preview: (
        <div className="bg-white rounded-lg p-4 h-32 border border-gray-200 relative overflow-hidden">
          <div className="grid grid-cols-2 gap-2 h-full">
            <div className="text-center">
              <div className="text-lg font-bold text-green-600 mb-1">19.84%</div>
              <div className="text-xs text-gray-600">Expected Return</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-blue-600 mb-1">14.22%</div>
              <div className="text-xs text-gray-600">Volatility</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-orange-600 mb-1">1.395</div>
              <div className="text-xs text-gray-600">Sharpe Ratio</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-green-600 mb-1">24.14%</div>
              <div className="text-xs text-gray-600">Max Drawdown</div>
            </div>
          </div>
          <div className="absolute top-1 right-1 text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
            ✓ Optimized
          </div>
        </div>
      )
    },
    {
      icon: Shield,
      title: "Risk Management",
      description: "Multi-layered constraint systems ensure your portfolio stays within your risk tolerance and investment goals.",
      preview: (
        <div className="bg-white rounded-lg p-4 h-32 border border-gray-200 relative">
          <div className="text-xs text-gray-600 mb-2">Risk Tolerance</div>
          <div className="space-y-2">
            <div className="border border-gray-200 rounded px-3 py-1 text-xs text-gray-600">Conservative</div>
            <div className="border-2 border-blue-500 bg-blue-50 rounded px-3 py-1 text-xs text-blue-700 font-medium">Moderate</div>
            <div className="border border-gray-200 rounded px-3 py-1 text-xs text-gray-600">Aggressive</div>
          </div>
          <div className="absolute bottom-2 right-2 flex items-center space-x-1">
            <Shield className="h-4 w-4 text-blue-500" />
            <span className="text-xs text-blue-600 font-medium">Balanced</span>
          </div>
        </div>
      )
    },
    {
      icon: Target,
      title: "Custom Optimization",
      description: "Build your own portfolio and let our algorithms optimize it, or start fresh with AI-generated allocations.",
      preview: (
        <div className="bg-white rounded-lg p-4 h-32 border border-gray-200 relative">
          <div className="flex items-center h-full">
            <div className="w-16 h-16 rounded-full bg-gradient-to-r from-blue-500 via-green-500 via-red-500 via-orange-500 to-purple-500 relative">
              <div className="absolute inset-2 bg-white rounded-full flex items-center justify-center">
                <PieChart className="h-6 w-6 text-gray-600" />
              </div>
            </div>
            <div className="ml-4 flex-1">
              <div className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">MSFT</span>
                  <span className="font-medium">20.0%</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">GLD</span>
                  <span className="font-medium">20.0%</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-600">GS</span>
                  <span className="font-medium">15.2%</span>
                </div>
                <div className="text-xs text-gray-500">+5 more assets</div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      icon: BarChart3,
      title: "Real-Time Analytics",
      description: "Live performance tracking, efficient frontier analysis, and comprehensive portfolio metrics dashboard.",
      preview: (
        <div className="bg-white rounded-lg p-4 h-32 border border-gray-200 relative">
          <div className="h-full flex flex-col">
            <div className="flex justify-between items-center mb-2">
              <div className="text-xs text-gray-600">Performance</div>
              <div className="text-xs font-medium text-blue-600">+7.8%</div>
            </div>
            <div className="flex-1 relative">
              <svg className="w-full h-full" viewBox="0 0 120 40">
                <polyline
                  points="0,35 20,30 40,25 60,20 80,15 100,10 120,8"
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2"
                />
                <polyline
                  points="0,38 20,36 40,34 60,32 80,30 100,28 120,26"
                  fill="none"
                  stroke="#9ca3af"
                  strokeWidth="1"
                  strokeDasharray="2,2"
                />
              </svg>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Your Portfolio</span>
              <span>S&P 500</span>
            </div>
          </div>
        </div>
      )
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

  const technologies = [
    { name: "Python", icon: PythonIcon, description: "ML & Analytics" },
    { name: "React", icon: ReactIcon, description: "Frontend UI" },
    { name: "FastAPI", icon: FastAPIIcon, description: "Backend API" },
    { name: "PostgreSQL", icon: PostgreSQLIcon, description: "Database" }
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
                  <p className="text-gray-600 leading-relaxed mb-6">{feature.description}</p>
                  
                  {/* Feature Preview */}
                  <div className="relative">
                    <div className="text-sm text-gray-500 mb-2 font-medium">Preview:</div>
                    {feature.preview}
                  </div>
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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            <div>
              <div className="flex items-center mb-4">
                <div className="h-8 w-8 bg-blue-600 rounded-lg flex items-center justify-center mr-3">
                  <TrendingUp className="h-5 w-5 text-white" />
                </div>
                <span className="text-xl font-bold">PortfolioMax</span>
              </div>
              <p className="text-gray-400 mb-8">
                AI-powered portfolio optimization for the modern investor.
              </p>
            </div>
            
            <div>
              <h4 className="font-semibold mb-6 text-lg">Built With</h4>
              <div className="grid grid-cols-2 gap-6">
                {technologies.map((tech, index) => {
                  const IconComponent = tech.icon;
                  return (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1, duration: 0.4 }}
                      className="flex items-center space-x-3 group hover:scale-105 transition-transform duration-200"
                    >
                      <div className="flex-shrink-0">
                        <IconComponent className="h-8 w-8" />
                      </div>
                      <div>
                        <div className="font-medium text-white group-hover:text-blue-300 transition-colors">
                          {tech.name}
                        </div>
                        <div className="text-sm text-gray-400">
                          {tech.description}
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
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