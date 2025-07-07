import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Target, 
  Shield, 
  TrendingUp, 
  Settings, 
  Info,
  Play,
  Zap
} from 'lucide-react';

const OptimizationControls = ({ 
  onOptimize, 
  onLoadEfficientFrontier, 
  isLoading, 
  stocks, 
  clusters 
}) => {
  const [formData, setFormData] = useState({
    riskTolerance: 'moderate',
    optimizationObjective: 'max_sharpe',
    constraintLevel: 'enhanced',
    returnMethod: 'ml_enhanced',
    targetReturn: '',
    portfolioValue: 100000,
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const riskToleranceOptions = [
    { 
      value: 'conservative', 
      label: 'Conservative', 
      description: 'Lower risk, stable returns',
      color: 'text-green-600',
      bgColor: 'bg-green-50 border-green-200'
    },
    { 
      value: 'moderate', 
      label: 'Moderate', 
      description: 'Balanced risk and return',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 border-blue-200'
    },
    { 
      value: 'aggressive', 
      label: 'Aggressive', 
      description: 'Higher risk, potential for higher returns',
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 border-purple-200'
    },
  ];

  const objectiveOptions = [
    { 
      value: 'max_sharpe', 
      label: 'Maximize Sharpe Ratio', 
      description: 'Best risk-adjusted returns',
      icon: Target
    },
    { 
      value: 'min_volatility', 
      label: 'Minimize Volatility', 
      description: 'Lowest risk portfolio',
      icon: Shield
    },
  ];

  const constraintOptions = [
    { value: 'basic', label: 'Basic', description: 'Simple constraints' },
    { value: 'enhanced', label: 'Enhanced', description: 'Recommended constraints' },
    { value: 'strict', label: 'Strict', description: 'Maximum diversification' },
  ];

  const returnMethodOptions = [
    { value: 'historical', label: 'Historical', description: 'Based on past performance' },
    { value: 'ml_enhanced', label: 'ML Enhanced', description: 'Machine learning predictions (Recommended)' },
    { value: 'conservative', label: 'Conservative', description: 'Conservative estimates' },
    { value: 'risk_adjusted', label: 'Risk Adjusted', description: 'Risk-adjusted estimates' },
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const optimizationParams = {
      risk_tolerance: formData.riskTolerance,
      optimization_objective: formData.optimizationObjective,
      constraint_level: formData.constraintLevel,
      return_method: formData.returnMethod,
      ...(formData.targetReturn && { target_return: parseFloat(formData.targetReturn) / 100 }),
    };

    onOptimize(optimizationParams, formData.portfolioValue);
  };

  const handleEfficientFrontier = () => {
    onLoadEfficientFrontier(formData.constraintLevel, formData.returnMethod);
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <div className="space-y-8">
      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card text-center">
          <div className="text-2xl font-bold text-blue-600">{stocks.length}</div>
          <div className="text-sm text-gray-600">Available Stocks</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-purple-600">{clusters.length}</div>
          <div className="text-sm text-gray-600">Asset Clusters</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-green-600">
            ${formData.portfolioValue.toLocaleString()}
          </div>
          <div className="text-sm text-gray-600">Portfolio Value</div>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Risk Tolerance */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Shield className="h-5 w-5 mr-2 text-blue-600" />
            Risk Tolerance
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {riskToleranceOptions.map((option) => (
              <motion.label
                key={option.value}
                whileHover={{ scale: 1.02 }}
                className={`
                  cursor-pointer rounded-lg border-2 p-4 transition-all duration-200
                  ${formData.riskTolerance === option.value 
                    ? `${option.bgColor} border-current ${option.color}` 
                    : 'bg-white border-gray-200 hover:border-gray-300'
                  }
                `}
              >
                <input
                  type="radio"
                  name="riskTolerance"
                  value={option.value}
                  checked={formData.riskTolerance === option.value}
                  onChange={(e) => handleChange('riskTolerance', e.target.value)}
                  className="sr-only"
                />
                <div className="text-center">
                  <div className={`font-semibold ${option.color}`}>
                    {option.label}
                  </div>
                  <div className="text-sm text-gray-600 mt-1">
                    {option.description}
                  </div>
                </div>
              </motion.label>
            ))}
          </div>
        </div>

        {/* Optimization Objective */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Target className="h-5 w-5 mr-2 text-blue-600" />
            Optimization Objective
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {objectiveOptions.map((option) => {
              const Icon = option.icon;
              return (
                <motion.label
                  key={option.value}
                  whileHover={{ scale: 1.02 }}
                  className={`
                    cursor-pointer rounded-lg border-2 p-4 transition-all duration-200
                    ${formData.optimizationObjective === option.value 
                      ? 'bg-blue-50 border-blue-500 text-blue-600' 
                      : 'bg-white border-gray-200 hover:border-gray-300'
                    }
                  `}
                >
                  <input
                    type="radio"
                    name="optimizationObjective"
                    value={option.value}
                    checked={formData.optimizationObjective === option.value}
                    onChange={(e) => handleChange('optimizationObjective', e.target.value)}
                    className="sr-only"
                  />
                  <div className="flex items-center">
                    <Icon className="h-6 w-6 mr-3" />
                    <div>
                      <div className="font-semibold">{option.label}</div>
                      <div className="text-sm text-gray-600">{option.description}</div>
                    </div>
                  </div>
                </motion.label>
              );
            })}
          </div>
        </div>

        {/* Portfolio Value */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Portfolio Value
          </h3>
          
          <div className="max-w-md">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Investment Amount (USD)
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <span className="text-gray-500">$</span>
              </div>
              <input
                type="number"
                value={formData.portfolioValue}
                onChange={(e) => handleChange('portfolioValue', parseInt(e.target.value) || 0)}
                className="input-field pl-8"
                min="1000"
                step="1000"
                placeholder="100000"
              />
            </div>
          </div>
        </div>

        {/* Target Return (Optional) */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2 text-blue-600" />
            Target Return (Optional)
            <div className="ml-2 relative group">
              <Info className="h-4 w-4 text-gray-400 cursor-help" />
              <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block">
                <div className="bg-gray-800 text-white text-xs rounded py-1 px-2 whitespace-nowrap">
                  Leave empty to optimize without target return
                </div>
              </div>
            </div>
          </h3>
          
          <div className="max-w-md">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Annual Target Return (%)
            </label>
            <input
              type="number"
              value={formData.targetReturn}
              onChange={(e) => handleChange('targetReturn', e.target.value)}
              className="input-field"
              min="0"
              max="100"
              step="0.1"
              placeholder="e.g., 12.5"
            />
            {formData.targetReturn && (
              <p className="text-sm text-gray-600 mt-1">
                Target: {formData.targetReturn}% annual return
              </p>
            )}
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="card">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center justify-between w-full text-left"
          >
            <h3 className="text-lg font-semibold text-gray-900 flex items-center">
              <Settings className="h-5 w-5 mr-2 text-blue-600" />
              Advanced Settings
            </h3>
            <motion.div
              animate={{ rotate: showAdvanced ? 180 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <svg className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </motion.div>
          </button>

          <motion.div
            initial={false}
            animate={{ height: showAdvanced ? 'auto' : 0, opacity: showAdvanced ? 1 : 0 }}
            transition={{ duration: 0.3 }}
            style={{ overflow: 'hidden' }}
          >
            <div className="pt-6 space-y-6">
              {/* Constraint Level */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Constraint Level
                </label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {constraintOptions.map((option) => (
                    <label
                      key={option.value}
                      className={`
                        cursor-pointer rounded-lg border p-3 text-center transition-all duration-200
                        ${formData.constraintLevel === option.value 
                          ? 'bg-blue-50 border-blue-500 text-blue-600' 
                          : 'bg-white border-gray-200 hover:border-gray-300'
                        }
                      `}
                    >
                      <input
                        type="radio"
                        name="constraintLevel"
                        value={option.value}
                        checked={formData.constraintLevel === option.value}
                        onChange={(e) => handleChange('constraintLevel', e.target.value)}
                        className="sr-only"
                      />
                      <div className="font-medium">{option.label}</div>
                      <div className="text-xs text-gray-600">{option.description}</div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Return Method */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Return Estimation Method
                </label>
                <div className="space-y-2">
                  {returnMethodOptions.map((option) => (
                    <label
                      key={option.value}
                      className={`
                        cursor-pointer rounded-lg border p-3 flex items-center transition-all duration-200
                        ${formData.returnMethod === option.value 
                          ? 'bg-blue-50 border-blue-500' 
                          : 'bg-white border-gray-200 hover:border-gray-300'
                        }
                      `}
                    >
                      <input
                        type="radio"
                        name="returnMethod"
                        value={option.value}
                        checked={formData.returnMethod === option.value}
                        onChange={(e) => handleChange('returnMethod', e.target.value)}
                        className="sr-only"
                      />
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">{option.label}</div>
                        <div className="text-sm text-gray-600">{option.description}</div>
                      </div>
                      {option.value === 'ml_enhanced' && (
                        <span className="bg-green-100 text-green-800 text-xs px-2 py-1 rounded-full ml-2">
                          Recommended
                        </span>
                      )}
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type="submit"
            disabled={isLoading}
            className="btn-primary flex-1 py-3 text-lg font-semibold flex items-center justify-center"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Optimizing Portfolio...
              </>
            ) : (
              <>
                <Play className="h-5 w-5 mr-2" />
                Optimize Portfolio
              </>
            )}
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type="button"
            onClick={handleEfficientFrontier}
            className="btn-secondary flex-1 py-3 text-lg font-semibold flex items-center justify-center"
          >
            <Zap className="h-5 w-5 mr-2" />
            Load Efficient Frontier
          </motion.button>
        </div>
      </form>
    </div>
  );
};

export default OptimizationControls; 