import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, TrendingUp, Shield } from 'lucide-react';

const StockClusterView = ({ stocks, clusters }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCluster, setSelectedCluster] = useState('all');
  const [sortBy, setSortBy] = useState('sharpe_ratio');

  // Filter and sort stocks
  const filteredStocks = stocks
    .filter(stock => 
      stock.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      stock.sector.toLowerCase().includes(searchTerm.toLowerCase())
    )
    .filter(stock => 
      selectedCluster === 'all' || stock.cluster.toString() === selectedCluster
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'sharpe_ratio':
          return b.sharpe_ratio - a.sharpe_ratio;
        case 'expected_return':
          return b.expected_return - a.expected_return;
        case 'volatility':
          return a.volatility - b.volatility;
        case 'symbol':
          return a.symbol.localeCompare(b.symbol);
        default:
          return 0;
      }
    });

  const getClusterColor = (clusterId) => {
    const colors = ['bg-blue-100 text-blue-800', 'bg-red-100 text-red-800', 'bg-green-100 text-green-800', 'bg-yellow-100 text-yellow-800', 'bg-purple-100 text-purple-800'];
    return colors[clusterId % colors.length];
  };

  const formatPercentage = (value) => `${(value * 100).toFixed(2)}%`;

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search stocks or sectors..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input-field pl-10"
          />
        </div>

        {/* Cluster Filter */}
        <select
          value={selectedCluster}
          onChange={(e) => setSelectedCluster(e.target.value)}
          className="input-field"
        >
          <option value="all">All Clusters</option>
          {clusters.map(cluster => (
            <option key={cluster.cluster_id} value={cluster.cluster_id.toString()}>
              {cluster.cluster_name}
            </option>
          ))}
        </select>

        {/* Sort */}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="input-field"
        >
          <option value="sharpe_ratio">Sort by Sharpe Ratio</option>
          <option value="expected_return">Sort by Expected Return</option>
          <option value="volatility">Sort by Volatility</option>
          <option value="symbol">Sort by Symbol</option>
        </select>
      </div>

      {/* Cluster Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {clusters.map((cluster) => (
          <motion.div
            key={cluster.cluster_id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card"
          >
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-gray-900">{cluster.cluster_name}</h4>
              <span className={`px-2 py-1 rounded-full text-xs ${getClusterColor(cluster.cluster_id)}`}>
                Cluster {cluster.cluster_id}
              </span>
            </div>
            <p className="text-sm text-gray-600 mb-3">{cluster.description}</p>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Assets:</span>
                <span className="font-medium">{cluster.asset_count}</span>
              </div>
              <div className="flex justify-between">
                <span>Avg Sharpe:</span>
                <span className="font-medium">{cluster.avg_sharpe.toFixed(2)}</span>
              </div>
              <div className="flex justify-between">
                <span>Avg Volatility:</span>
                <span className="font-medium">{formatPercentage(cluster.avg_volatility)}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Stock Table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Available Stocks ({filteredStocks.length})
          </h3>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sector
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Cluster
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sharpe Ratio
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Expected Return
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Volatility
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredStocks.map((stock, index) => (
                <motion.tr
                  key={stock.symbol}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.02 }}
                  className="hover:bg-gray-50"
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-8 w-8">
                        <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center">
                          <span className="text-xs font-medium text-blue-600">
                            {stock.symbol.slice(0, 2)}
                          </span>
                        </div>
                      </div>
                      <div className="ml-3">
                        <div className="text-sm font-medium text-gray-900">
                          {stock.symbol}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {stock.sector}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 py-1 rounded-full text-xs ${getClusterColor(stock.cluster)}`}>
                      {stock.cluster_name}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <div className="flex items-center justify-end">
                      <TrendingUp className="h-4 w-4 text-green-500 mr-1" />
                      <span className="text-sm font-medium text-gray-900">
                        {stock.sharpe_ratio.toFixed(2)}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900">
                    {formatPercentage(stock.expected_return)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <div className="flex items-center justify-end">
                      <Shield className="h-4 w-4 text-blue-500 mr-1" />
                      <span className="text-sm text-gray-900">
                        {formatPercentage(stock.volatility)}
                      </span>
                    </div>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {filteredStocks.length === 0 && (
          <div className="text-center py-8">
            <p className="text-gray-500">No stocks match your search criteria</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockClusterView; 