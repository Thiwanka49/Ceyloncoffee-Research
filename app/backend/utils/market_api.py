import requests
from datetime import datetime
import os

# ============================================================================
# EXCHANGE RATE API
# ============================================================================

def get_usd_lkr_rate():
    """
    Get current USD to LKR exchange rate
    Using exchangerate-api.com (free tier: 1,500 requests/month)
    """
    try:
        # Free API - no key needed for basic usage
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'rates' in data and 'LKR' in data['rates']:
            rate = data['rates']['LKR']
            print(f"✅ USD to LKR rate fetched: {rate}")
            return {
                'success': True,
                'rate': round(rate, 2),
                'date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'source': 'exchangerate-api.com'
            }
        else:
            return {
                'success': False,
                'error': 'LKR rate not found in response'
            }
            
    except Exception as e:
        print(f"❌ Error fetching exchange rate: {e}")
        return {
            'success': False,
            'error': str(e)
        }


# ============================================================================
# COFFEE PRICE API
# ============================================================================

def get_global_coffee_price():
    """
    Get global coffee price (Arabica)
    Returns estimated price based on recent market trends.
    """
    try:
        # Average Arabica C-Market price estimate
        # Current market (2025-2026): ~$2.00-2.50 per lb = ~$4.40-5.50 per kg
        
        estimated_price_usd_per_kg = 4.80  # Conservative middle estimate
        
        print(f"✅ Coffee price estimate: ${estimated_price_usd_per_kg}/kg")
        
        return {
            'success': True,
            'price_usd_per_kg': estimated_price_usd_per_kg,
            'note': 'Estimated based on recent Arabica market trends. For precise pricing, check ICO (International Coffee Organization) or commodity exchanges.',
            'source': 'Market estimate',
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        print(f"❌ Error fetching coffee price: {e}")
        return {
            'success': False,
            'error': str(e)
        }