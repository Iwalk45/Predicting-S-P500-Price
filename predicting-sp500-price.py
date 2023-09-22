{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "73ab7299",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from yahoo_fin import options as ops\n",
    "import yfinance as yf\n",
    "from scipy.stats import norm\n",
    "from datetime import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "b6f693c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "SPY = ops.get_calls('spy','2023-12-29')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "312cc336",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Contract Name</th>\n",
       "      <th>Last Trade Date</th>\n",
       "      <th>Strike</th>\n",
       "      <th>Last Price</th>\n",
       "      <th>Bid</th>\n",
       "      <th>Ask</th>\n",
       "      <th>Change</th>\n",
       "      <th>% Change</th>\n",
       "      <th>Volume</th>\n",
       "      <th>Open Interest</th>\n",
       "      <th>Implied Volatility</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>SPY231229C00270000</td>\n",
       "      <td>2023-09-21 3:33PM EDT</td>\n",
       "      <td>270.0</td>\n",
       "      <td>165.29</td>\n",
       "      <td>164.10</td>\n",
       "      <td>165.19</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-</td>\n",
       "      <td>2</td>\n",
       "      <td>62</td>\n",
       "      <td>61.60%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>SPY231229C00275000</td>\n",
       "      <td>2023-08-04 11:02AM EDT</td>\n",
       "      <td>275.0</td>\n",
       "      <td>180.85</td>\n",
       "      <td>178.60</td>\n",
       "      <td>179.77</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-</td>\n",
       "      <td>2</td>\n",
       "      <td>59</td>\n",
       "      <td>108.53%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>SPY231229C00280000</td>\n",
       "      <td>2023-08-10 3:09PM EDT</td>\n",
       "      <td>280.0</td>\n",
       "      <td>171.35</td>\n",
       "      <td>168.33</td>\n",
       "      <td>169.43</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-</td>\n",
       "      <td>1</td>\n",
       "      <td>50</td>\n",
       "      <td>94.80%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>SPY231229C00285000</td>\n",
       "      <td>2023-08-01 1:08PM EDT</td>\n",
       "      <td>285.0</td>\n",
       "      <td>176.60</td>\n",
       "      <td>168.41</td>\n",
       "      <td>169.49</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-</td>\n",
       "      <td>2</td>\n",
       "      <td>17</td>\n",
       "      <td>102.11%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>SPY231229C00290000</td>\n",
       "      <td>2023-08-21 2:16PM EDT</td>\n",
       "      <td>290.0</td>\n",
       "      <td>152.98</td>\n",
       "      <td>151.89</td>\n",
       "      <td>153.09</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-</td>\n",
       "      <td>3</td>\n",
       "      <td>16</td>\n",
       "      <td>75.68%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110</th>\n",
       "      <td>SPY231229C00525000</td>\n",
       "      <td>2023-09-22 10:14AM EDT</td>\n",
       "      <td>525.0</td>\n",
       "      <td>0.04</td>\n",
       "      <td>0.04</td>\n",
       "      <td>0.06</td>\n",
       "      <td>-0.02</td>\n",
       "      <td>-33.33%</td>\n",
       "      <td>4</td>\n",
       "      <td>949</td>\n",
       "      <td>14.75%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>111</th>\n",
       "      <td>SPY231229C00530000</td>\n",
       "      <td>2023-09-22 10:27AM EDT</td>\n",
       "      <td>530.0</td>\n",
       "      <td>0.04</td>\n",
       "      <td>0.04</td>\n",
       "      <td>0.05</td>\n",
       "      <td>0.01</td>\n",
       "      <td>+33.33%</td>\n",
       "      <td>12</td>\n",
       "      <td>5985</td>\n",
       "      <td>15.09%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>112</th>\n",
       "      <td>SPY231229C00535000</td>\n",
       "      <td>2023-09-20 3:23PM EDT</td>\n",
       "      <td>535.0</td>\n",
       "      <td>0.05</td>\n",
       "      <td>0.03</td>\n",
       "      <td>0.04</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-</td>\n",
       "      <td>10</td>\n",
       "      <td>244</td>\n",
       "      <td>15.33%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>113</th>\n",
       "      <td>SPY231229C00540000</td>\n",
       "      <td>2023-09-19 9:42AM EDT</td>\n",
       "      <td>540.0</td>\n",
       "      <td>0.03</td>\n",
       "      <td>0.02</td>\n",
       "      <td>0.04</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-</td>\n",
       "      <td>2</td>\n",
       "      <td>673</td>\n",
       "      <td>15.92%</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>114</th>\n",
       "      <td>SPY231229C00550000</td>\n",
       "      <td>2023-09-22 9:49AM EDT</td>\n",
       "      <td>550.0</td>\n",
       "      <td>0.03</td>\n",
       "      <td>0.02</td>\n",
       "      <td>0.03</td>\n",
       "      <td>0.00</td>\n",
       "      <td>-</td>\n",
       "      <td>1</td>\n",
       "      <td>2727</td>\n",
       "      <td>16.60%</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>115 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "          Contract Name         Last Trade Date  Strike  Last Price     Bid  \\\n",
       "0    SPY231229C00270000   2023-09-21 3:33PM EDT   270.0      165.29  164.10   \n",
       "1    SPY231229C00275000  2023-08-04 11:02AM EDT   275.0      180.85  178.60   \n",
       "2    SPY231229C00280000   2023-08-10 3:09PM EDT   280.0      171.35  168.33   \n",
       "3    SPY231229C00285000   2023-08-01 1:08PM EDT   285.0      176.60  168.41   \n",
       "4    SPY231229C00290000   2023-08-21 2:16PM EDT   290.0      152.98  151.89   \n",
       "..                  ...                     ...     ...         ...     ...   \n",
       "110  SPY231229C00525000  2023-09-22 10:14AM EDT   525.0        0.04    0.04   \n",
       "111  SPY231229C00530000  2023-09-22 10:27AM EDT   530.0        0.04    0.04   \n",
       "112  SPY231229C00535000   2023-09-20 3:23PM EDT   535.0        0.05    0.03   \n",
       "113  SPY231229C00540000   2023-09-19 9:42AM EDT   540.0        0.03    0.02   \n",
       "114  SPY231229C00550000   2023-09-22 9:49AM EDT   550.0        0.03    0.02   \n",
       "\n",
       "        Ask  Change % Change  Volume  Open Interest Implied Volatility  \n",
       "0    165.19    0.00        -       2             62             61.60%  \n",
       "1    179.77    0.00        -       2             59            108.53%  \n",
       "2    169.43    0.00        -       1             50             94.80%  \n",
       "3    169.49    0.00        -       2             17            102.11%  \n",
       "4    153.09    0.00        -       3             16             75.68%  \n",
       "..      ...     ...      ...     ...            ...                ...  \n",
       "110    0.06   -0.02  -33.33%       4            949             14.75%  \n",
       "111    0.05    0.01  +33.33%      12           5985             15.09%  \n",
       "112    0.04    0.00        -      10            244             15.33%  \n",
       "113    0.04    0.00        -       2            673             15.92%  \n",
       "114    0.03    0.00        -       1           2727             16.60%  \n",
       "\n",
       "[115 rows x 11 columns]"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "SPY"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5f74aa15",
   "metadata": {},
   "outputs": [],
   "source": [
    "def call_price(S,K,T,r,vol):\n",
    "    d_1 = (np.log(S/K)+(r+(vol**2)/2)*T)/(vol*np.sqrt(T))\n",
    "    d_2 = d_1 - vol*np.sqrt(T)\n",
    "    return S*norm.cdf(d_1) - K*np.exp(-r*T)*norm.cdf(d_2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9dde46fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "def vega(S,K,T,r,vol):\n",
    "    d_1 = (np.log(S/K)+(r+(vol**2)/2)*T)/(vol*np.sqrt(T))\n",
    "    return S*np.sqrt(T)*norm.cdf(d_1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "15f30f20",
   "metadata": {},
   "outputs": [],
   "source": [
    "def implied_vol(C,S,K,T,r):\n",
    "    vol = 0.6\n",
    "    vols = [vol]\n",
    "    call_price_0 = [np.abs(call_price(S,K,T,r,vol)-C)]\n",
    "    \n",
    "    for n in range(0,2000):\n",
    "        vol_new = vol - (call_price(S,K,T,r,vol)-C)/vega(S,K,T,r,vol)\n",
    "        vols.append(vol_new)\n",
    "        call_price_0.append(np.abs(call_price(S,K,T,r,vol)-C))\n",
    "        vol = vol_new\n",
    "        \n",
    "    return  vols[np.array(call_price_0).argmin()]\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "6ef04aae",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[*********************100%%**********************]  1 of 1 completed\n"
     ]
    }
   ],
   "source": [
    "SPY_data = yf.download('SPY','2023-01-01')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "07a451cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "def mu(S,T,r,vol):\n",
    "    return np.log(S)+(r-(vol**2)/2)*T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "id": "a4ba4853",
   "metadata": {},
   "outputs": [],
   "source": [
    "def lognormal_mean(S,T,r,vol):\n",
    "    average = mu(S,T,r,vol)\n",
    "    return np.exp(average+(vol**2)*T/2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "6cb3b68d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def lognormal_std(S,T,r,vol):\n",
    "    average = mu(S,T,r,vol)\n",
    "    mean = lognormal_mean(S,T,r,vol)\n",
    "    return mean*np.sqrt(np.exp((vol**2)/2)-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "46637ccb",
   "metadata": {},
   "outputs": [],
   "source": [
    "SPY['Last Trade Date'] = pd.to_datetime(SPY['Last Trade Date'])\n",
    "\n",
    "def get_spy_price(a):\n",
    "    return SPY_data.loc[a.normalize()]['Adj Close']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "f8b6b3da",
   "metadata": {},
   "outputs": [],
   "source": [
    "def time_difference(a):\n",
    "    time = datetime(2023,12,29) - a\n",
    "    return time.days/365"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "03861b69",
   "metadata": {},
   "outputs": [],
   "source": [
    "SPY['Time until maturity'] = SPY['Last Trade Date'].apply(time_difference)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "id": "a6c774d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\User\\AppData\\Local\\Temp\\ipykernel_8408\\1883529817.py:7: RuntimeWarning: divide by zero encountered in scalar divide\n",
      "  vol_new = vol - (call_price(S,K,T,r,vol)-C)/vega(S,K,T,r,vol)\n",
      "C:\\Users\\User\\AppData\\Local\\Temp\\ipykernel_8408\\1416521926.py:2: RuntimeWarning: invalid value encountered in scalar divide\n",
      "  d_1 = (np.log(S/K)+(r+(vol**2)/2)*T)/(vol*np.sqrt(T))\n"
     ]
    }
   ],
   "source": [
    "SPY['Implied volatility'] = np.vectorize(implied_vol)(SPY['Last Price'],SPY['Last Trade Date'].apply(get_spy_price),SPY['Strike'],SPY['Last Trade Date'].apply(time_difference),0.0533)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "b284d8ce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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hvxcd9xxBxYuXCg7PmbMGMyaNQvPP/88fvWrX8Hj8eCGG27Anj17kJaWFtJzBHunoaysDMXFxSFdMxrk4zWVW3f81TDoExEREYUlKYN+amqq72OXyxX0fKezpWc81AAOALm5uZpfnzdvHr777ju8+OKLOHr0KN555x1ce+21IT1HsPajeCHfGVepdUda0WfrDhEREVE4knLqTlZWlu9jPe049fX1APS1+YRj3rx5vo9XrFgRleeIB049rTucukNERERkiKQM+qmpqWjfvj2A4BNrqqqqfEE/WoteBw0a5Pv4yJEjUXmOeCAfrxm8daeOQZ+IiIgoLEkZ9AH4drzds2cP3G71MLljxw7ZY4wmimJUrhtvZK07DnlFP1O6GJdTd4iIiIjCkrRBf/z48QBa2nLWr1+vep5/K824ceOici/btm3zfdylS5eoPEc8kE3dUdgwK5utO0RERESGSNqgf9lll/k+XrRokeI5Xq8XS5YsAdCyqHbSpElRuZfnn3/e97H/Trxtja7xmrLFuAz6REREROFI2qBfXFyMCRMmAABeeuklrFmzRnbOI4884tsN9/bbb4fdHlhtXrx4MQRBgCAIiiM0t2zZgj179mjex/PPP4+XXnoJANCpUydcfvnl4Xw7CaFRNnVHqUefFX0iIiIiIyTleM1WTzzxBMaNG4fGxkZMmTIFCxYswKRJk9DY2IjXX38dL7zwAgCgX79+mD9/fsjXX79+PW644QZMmjQJP/nJTzB06FC0a9cObrcbO3bswKuvvorPPvsMAGC1WvH8888jIyPD0O8xnuiZupPJ8ZpEREREhkjqoD9ixAi88cYbuO6661BTU4MFCxbIzunXrx+WLVsWMJIzFB6PB59//jk+//xz1XPatWuHl156CdOmTQvrORKFvHUn+NQdp9sLl9sLh8KEHiIiIiJSl9RBHwAuueQSbN68GU888QSWLVuG0tJSOBwOFBUVYfr06bjllluQnp4e1rV/+tOf+tqCvv/+exw/fhwnT56EKIrIz8/HGWecgYsuughz5sxBdna2wd9Z/Glyh75hFtBS1W+XmRK1+yIiIiJqiwQxWWY7JrHS0lLfHgCHDx+O2U66g+79GA2u01X9d246GyN75AWc09TswYD/+zjg2PLfnIte7dtuSxMRERFRNPIa+yHIFKIo6hqvmWKzwG4VAo7VObkgl4iI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7bc4dOgQgoOD8fTTT7fr+yAi0kYhOvqUChERdXqVlZXo1auX3uvD9+nTB//+978xZMgQtddLS0tx33334cqVKxr7+Pr64urVq9L39VcCYmNjsXLlymaPt23bNunMe3Z2tsYDtiIiIpCYmIjw8HAkJCRo7J+dnY3w8HDk5OTA2toaX331FaKioqTy3377DXPmzMGRI0ea7QcAREZG4tixYzrrERG1NU7dISIinaytrXH9+nWcOnUKb7zxBsaPHw9/f3/Y2dnB3Nwc3bp1Q2BgIGbPno0dO3bg/PnzGkk+ANjb2yMpKQnPP/88goKCYGtr2wHvRlOvXr0QHx8Pb29vVFZWYvLkyWrJurOzMw4fPozvvvsO0dHR6NOnD+zt7aFUKuHs7IyhQ4fi2WefxaFDh3D06NEOfCdERL/jGX0iIiIiIhniGX0iIiIiIhliok9EREREJENM9ImIiIiIZIiJPhERERGRDDHRJyIiIiKSISb6REREREQyxESfiIiIiEiGmOgTEREREckQE30iIiIiIhliok9EREREJENM9ImIiIiIZIiJPhERERGRDDHRJyIiIiKSISb6REREREQyxESfiIiIiEiGmOgTEREREckQE30iIiIiIhliok9EREREJENM9ImIiIiIZIiJPhERERGRDDHRJyIiIiKSISb6REREREQyxESfiIiIiEiGmOgTEREREcnQ/wOYh2Zcv8uSwQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 800x800 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure(figsize=(4,4),dpi=200)\n",
    "plt.plot(SPY.dropna()['Strike'],SPY.dropna()['Implied volatility'])\n",
    "plt.xlabel('Strike')\n",
    "plt.ylabel('Implied Volatility')\n",
    "plt.savefig('VolatilitySkew.png')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "947af664",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "id": "7cd6cc1e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\User\\AppData\\Local\\Temp\\ipykernel_8408\\1216655513.py:1: FutureWarning: DataFrame.mean and DataFrame.median with numeric_only=None will include datetime64 and datetime64tz columns in a future version.\n",
      "  SPY[(SPY['Strike'] > 400) & (SPY['Strike'] < 500)].mean()\n",
      "C:\\Users\\User\\AppData\\Local\\Temp\\ipykernel_8408\\1216655513.py:1: FutureWarning: The default value of numeric_only in DataFrame.mean is deprecated. In a future version, it will default to False. In addition, specifying 'numeric_only=None' is deprecated. Select only valid columns or specify the value of numeric_only to silence this warning.\n",
      "  SPY[(SPY['Strike'] > 400) & (SPY['Strike'] < 500)].mean()\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Strike                  450.000000\n",
       "Last Price               11.293158\n",
       "Bid                      10.887368\n",
       "Ask                      11.057368\n",
       "Change                   -0.553684\n",
       "Volume                  322.210526\n",
       "Open Interest          6242.842105\n",
       "Implied volatility        0.139542\n",
       "Time until maturity       0.266186\n",
       "dtype: float64"
      ]
     },
     "execution_count": 144,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "SPY[(SPY['Strike'] > 400) & (SPY['Strike'] < 500)].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "7f2aa4cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "43.377621129115646"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lognormal_std(431.99,100/365,0.055,0.139542)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "166abada",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "438.54872933888487"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lognormal_mean(431.99,100/365,0.055,0.139542)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "fb3009a9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "395.1711"
      ]
     },
     "execution_count": 158,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "438.5487-43.3776"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c56bbedc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9285fd52",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e6109949",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
