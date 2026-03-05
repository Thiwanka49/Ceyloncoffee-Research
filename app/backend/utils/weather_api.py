"""
Weather API Service - Open-Meteo Integration
Gets historical weather data for yield prediction
"""

import requests
from datetime import datetime, timedelta
from statistics import mean

# ============================================================================
# SRI LANKAN COFFEE REGIONS (Coordinates)
# ============================================================================

COFFEE_REGIONS = {
    'kandy': {
        'name': 'Kandy District',
        'lat': 7.2906,
        'lon': 80.6337,
        'elevation': 500
    },
    'nuwara_eliya': {
        'name': 'Nuwara Eliya District',
        'lat': 6.9497,
        'lon': 80.7891,
        'elevation': 1868
    },
    'badulla': {
        'name': 'Badulla District',
        'lat': 6.9934,
        'lon': 81.0550,
        'elevation': 670
    },
    'ratnapura': {
        'name': 'Ratnapura District',
        'lat': 6.6828,
        'lon': 80.4036,
        'elevation': 33
    },
    'matale': {
        'name': 'Matale District',
        'lat': 7.4675,
        'lon': 80.6234,
        'elevation': 364
    }
}


def get_historical_weather(region_key, months=12):
    """
    Get historical weather data for the past N months
    
    Args:
        region_key: Key from COFFEE_REGIONS dict
        months: Number of past months to analyze (default: 12)
        
    Returns:
        dict: Aggregated weather data suitable for yield prediction
    """
    
    if region_key not in COFFEE_REGIONS:
        region_key = 'kandy'  # Default to Kandy
    
    region = COFFEE_REGIONS[region_key]
    
    # # Calculate date range (past N months)
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=months*30)

    # Cap end date to 7 days ago to avoid archive delay / future issues
    today = datetime.now()
    safe_end_date = today - timedelta(days=7)  # or 5–10 days; adjust if needed
    end_date = safe_end_date
    
    # Start date: approx months back from safe end
    start_date = safe_end_date - timedelta(days=months * 30)
    
    print(f"\n🌤️  Fetching weather data for {region['name']}")
    print(f"   Period: {start_date.date()} to {end_date.date()}")
    
    try:
        # Open-Meteo Archive API
        url = 'https://archive-api.open-meteo.com/v1/archive'
        
        params = {
            'latitude': region['lat'],
            'longitude': region['lon'],
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'daily': [
                'temperature_2m_mean',
                'temperature_2m_max',
                'temperature_2m_min',
                'relative_humidity_2m_mean',
                'precipitation_sum',
                'rain_sum',
                'cloudcover_mean',
                'shortwave_radiation_sum'
            ],
            'timezone': 'Asia/Colombo'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        daily = data['daily']
        
        # Calculate aggregated statistics
        weather_stats = calculate_weather_statistics(daily, months)
        
        print(f"✅ Weather data retrieved successfully")
        print(f"   Avg Temp: {weather_stats['avg_temp']:.1f}°C")
        print(f"   Total Rainfall: {weather_stats['total_rainfall_mm']:.0f}mm")
        
        return {
            'success': True,
            'data': weather_stats,
            'region': region['name'],
            'period': f"{start_date.date()} to {end_date.date()}",
            'source': 'Open-Meteo Historical Weather API'
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return {
            'success': False,
            'error': f'Failed to fetch weather data: {str(e)}'
        }
    except Exception as e:
        print(f"❌ Processing Error: {e}")
        return {
            'success': False,
            'error': f'Failed to process weather data: {str(e)}'
        }


def calculate_weather_statistics(daily_data, months):
    """
    Calculate aggregated weather statistics from daily data
    """
    
    # Temperature statistics
    temps = [t for t in daily_data['temperature_2m_mean'] if t is not None]
    temp_max = [t for t in daily_data['temperature_2m_max'] if t is not None]
    temp_min = [t for t in daily_data['temperature_2m_min'] if t is not None]
    
    avg_temp = round(mean(temps), 1)
    max_temp = round(mean(temp_max), 1)
    min_temp = round(mean(temp_min), 1)
    
    # Humidity
    humidity = [h for h in daily_data['relative_humidity_2m_mean'] if h is not None]
    avg_humidity = round(mean(humidity), 1)
    
    # Rainfall
    rainfall = [r for r in daily_data['rain_sum'] if r is not None]
    total_rainfall = round(sum(rainfall), 1)
    rainy_days = len([r for r in rainfall if r > 1.0])  # Days with >1mm rain
    
    # Cloud cover
    cloudcover = [c for c in daily_data['cloudcover_mean'] if c is not None]
    avg_cloudcover = round(mean(cloudcover), 1)
    
    # Solar radiation (convert from MJ/m² to W/m²)
    # Open-Meteo gives shortwave_radiation_sum in MJ/m² per day
    # Average W/m² ≈ (MJ/m²/day) × 1000000 / 86400
    radiation = [r for r in daily_data['shortwave_radiation_sum'] if r is not None]
    avg_solar_mj = mean(radiation)
    avg_solarradiation = round(avg_solar_mj * 11.574, 1)  # Convert to W/m²
    
    # UV Index estimation (rough estimate based on solar radiation)
    # UV Index ≈ Solar Radiation (W/m²) / 25
    avg_uvindex = round(avg_solarradiation / 25, 1)
    
    # Ensure UV index is within realistic range
    avg_uvindex = max(5.0, min(11.0, avg_uvindex))
    
    return {
        'avg_temp': avg_temp,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'avg_humidity': avg_humidity,
        'total_rainfall_mm': total_rainfall,
        'rainy_days': rainy_days,
        'avg_cloudcover': avg_cloudcover,
        'avg_solarradiation': avg_solarradiation,
        'avg_uvindex': avg_uvindex
    }


def get_available_regions():
    """Get list of available coffee regions"""
    return {
        'regions': [
            {
                'key': key,
                'name': region['name'],
                'coordinates': {
                    'lat': region['lat'],
                    'lon': region['lon']
                },
                'elevation': region['elevation']
            }
            for key, region in COFFEE_REGIONS.items()
        ]
    }