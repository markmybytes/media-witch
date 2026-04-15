#!/usr/bin/env python3
from media_witch.features.torrent.decoder import bdecode
import sys
sys.path.insert(0, "src")


# Test a simple bencode string
test = b"d4:infod4:name9:valid.txt6:lengthi100eee"
try:
    result = bdecode(test)
    print("Success!")
    print(result)
except Exception as e:
    print(f"Error: {e}")
