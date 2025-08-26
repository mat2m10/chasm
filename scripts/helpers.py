def map_to_color(x, y, z, df, value):
    # Check if there's variance to avoid division by zero
    r = x / df['x'].max() if df['x'].max() != 0 else 0  # Red component based on 'x'
    g = y / df['y'].max() if df['y'].max() != 0 else 0  # Green component based on 'y'
    b = z / df[value].max() if df[value].max() != 0 else 0  # Blue component based on 'z'
    
    return (r, g, b)