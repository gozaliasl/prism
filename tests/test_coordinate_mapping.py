#!/usr/bin/env python3
"""
Test script for coordinate-to-tile mapping
"""

from shapely.geometry import Point, Polygon

def map_coordinates_to_tile(ra, dec):
    """
    Map COSMOS-Web coordinates to PSF tile names using exact tile boundaries
    
    Args:
        ra: Right Ascension in degrees
        dec: Declination in degrees
        
    Returns:
        str: Tile name (e.g., 'A1', 'B5') or None if not found
    """
    # COSMOS-Web tile boundaries (exact coordinates)
    coords_A1 = [(149.8703317, 2.0856512), (149.7198796, 2.1403395), (149.7908786, 2.3354095), (149.9413496, 2.2807163)]
    coords_A2 = [(150.0058959, 2.0363591), (149.8554506, 2.0910612), (149.9264667, 2.2861269), (150.0769300, 2.2314186)]
    coords_A3 = [(150.1414523, 1.9870553), (149.9910155, 2.0417704), (150.0620479, 2.2368306), (150.2125019, 2.1821081)]
    coords_A4 = [(150.2769995, 1.9377408), (150.1265729, 1.9924679), (150.1976208, 2.1875215), (150.3480637, 2.1327859)]
    coords_A5 = [(150.4125359, 1.8884166), (150.2621212, 1.9431545), (150.3331838, 2.1382005), (150.4836139, 2.0834528)]
    coords_A6 = [(149.8045087, 1.9048087), (149.6540746, 1.9594923), (149.7250552, 2.1545612), (149.8755087, 2.0998725)]
    coords_A7 = [(149.9400575, 1.8555218), (149.7896293, 1.9102182), (149.8606274, 2.1052826), (150.0110740, 2.0505800)]
    coords_A8 = [(150.0755992, 1.8062243), (149.9251788, 1.8609325), (149.9961935, 2.0559913), (150.1466316, 2.0012757)]
    coords_A9 = [(150.2111325, 1.7569171), (150.0607214, 1.8116361), (150.1317520, 2.0066883), (150.2821799, 1.9519607)]
    coords_A10= [(150.3466557, 1.7076011), (150.1962556, 1.7623299), (150.2673014, 1.9573744), (150.4177173, 1.9026358)]
    coords_B1 = [(150.0020274, 2.4473359), (149.8515406, 2.5020333), (149.9225757, 2.6970916), (150.0730806, 2.6423895)]
    coords_B2 = [(150.1376214, 2.3980335), (149.9871430, 2.4527469), (150.0581944, 2.6478011), (150.2086900, 2.5930817)]
    coords_B3 = [(150.2732061, 2.3487174), (150.1227378, 2.4034461), (150.1938048, 2.5984949), (150.3442894, 2.5437590)]
    coords_B4 = [(150.4087801, 2.2993886), (150.2583236, 2.3541315), (150.3294054, 2.5491739), (150.4798772, 2.4944226)]
    coords_B5 = [(150.5443418, 2.2500480), (150.3938989, 2.3048040), (150.4649946, 2.4998389), (150.6154520, 2.4450733)]
    coords_B6 = [(149.9361713, 2.2664951), (149.7857017, 2.3211879), (149.8567188, 2.5162544), (150.0072070, 2.4615567)]
    coords_B7 = [(150.0717506, 2.2171978), (149.9212885, 2.2719056), (149.9923224, 2.4669678), (150.1428020, 2.4122539)]
    coords_B8 = [(150.2073213, 2.1678878), (150.0568686, 2.2226097), (150.1279183, 2.4176665), (150.2783878, 2.3629373)]
    coords_B9 = [(150.3428821, 2.1185662), (150.1924404, 2.1733011), (150.2635052, 2.3683514), (150.4139629, 2.3136080)]
    coords_B10= [(150.4784314, 2.0692337), (150.3280023, 2.1239807), (150.3990815, 2.3190234), (150.5495255, 2.2642668)]
    
    # Create polygon objects
    polygons = {
        'A1': Polygon(coords_A1), 'A2': Polygon(coords_A2), 'A3': Polygon(coords_A3), 'A4': Polygon(coords_A4), 'A5': Polygon(coords_A5),
        'A6': Polygon(coords_A6), 'A7': Polygon(coords_A7), 'A8': Polygon(coords_A8), 'A9': Polygon(coords_A9), 'A10': Polygon(coords_A10),
        'B1': Polygon(coords_B1), 'B2': Polygon(coords_B2), 'B3': Polygon(coords_B3), 'B4': Polygon(coords_B4), 'B5': Polygon(coords_B5),
        'B6': Polygon(coords_B6), 'B7': Polygon(coords_B7), 'B8': Polygon(coords_B8), 'B9': Polygon(coords_B9), 'B10': Polygon(coords_B10)
    }
    
    # Create point from coordinates
    point = Point(ra, dec)
    
    # Check which tile contains the point
    for tile_name, polygon in polygons.items():
        if polygon.contains(point):
            return tile_name
    
    # If no tile contains the point, return None
    return None

if __name__ == "__main__":
    # Test with some sample coordinates
    test_coords = [
        (149.8575463238348, 2.093537453758908),  # From galaxy catalog
        (149.8614671527942, 2.09408385222106),   # From galaxy catalog
        (150.0, 2.0),  # Should be in A1
        (150.2, 2.3),  # Should be in B3
    ]

    print('Testing coordinate-to-tile mapping:')
    for ra, dec in test_coords:
        tile = map_coordinates_to_tile(ra, dec)
        print(f'RA={ra:.6f}, DEC={dec:.6f} -> {tile}')
