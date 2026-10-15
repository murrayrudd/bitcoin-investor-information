#include <cmath>
#include <algorithm>
extern "C" double bitcoin_excess(int n, double price, const double* cash,
 const double* holdings, const double* floor, const double* cap,
 const double* logodds, const double* invgamma, double low, double high, double stock) {
    double total=0.0;
    if(price>=high) return -stock;
    const double a=high-price, b=price-low;
    const double offset=price>low ? std::log(a/b) : 0.0;
    for(int i=0;i<n;++i){
        const double upper=std::max(0.0,std::min(cap[i],holdings[i]+(cash[i]-floor[i])/price));
        double x=upper;
        if(price>low){
            const double z=(logodds[i]+offset)*invgamma[i];
            const double s=z>=0 ? 1.0/(1.0+std::exp(-z)) : std::exp(z)/(1.0+std::exp(z));
            x=(cash[i]+price*holdings[i])*(2*s-1)/(a*(1-s)+b*s);
            x=std::max(0.0,std::min(upper,x));
        }
        total+=x;
    }
    return total-stock;
}
