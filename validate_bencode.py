#!/usr/bin/env python
"""Validate bencode test data."""

from media_witch.features.torrent.decoder import bdecode
import sys
sys.path.insert(0, 'src')


tests = [
    ("test_creates_multiple_files",
     b"d4:infod4:name4:Root5:filesl"
     b"d4:pathl9:file1.txte6:lengthi100eee"
     b"d4:pathl9:file2.txte6:lengthi200eee"
     b"d4:pathl9:file3.txte6:lengthi300eeee"
     b"ee"),

    ("test_creates_nested_directories",
     b"d4:infod4:name4:Root5:filesl"
     b"d4:pathl3:dir5:file1e6:lengthi100eee"
     b"d4:pathl3:dir4:sub25:file2e6:lengthi200eeee"
     b"ee"),

    ("test_multiple_files_same_directory",
     b"d4:infod4:name4:Root5:filesl"
     b"d4:pathl3:dir5:file1e6:lengthi100eee"
     b"d4:pathl3:dir5:file2e6:lengthi200eee"
     b"d4:pathl3:dir5:file3e6:lengthi300eeee"
     b"ee"),

    ("test_batch_continues_on_error",
     b"d4:infod4:name9:valid.txt6:lengthi100eee"),

    ("test_tv_show_season_structure",
     b"d4:infod4:name23:Show.Name.S01.1080p.WEB5:filesl"
     b"d4:pathl20:Show.Name.S01E01.mkve6:lengthi1500000000eee"
     b"d4:pathl20:Show.Name.S01E02.mkve6:lengthi1600000000eee"
     b"d4:pathl4:Subs23:Show.Name.S01E01.en.srte6:lengthi50000eee"
     b"d4:pathl4:Subs23:Show.Name.S01E02.en.srte6:lengthi52000eeee"
     b"ee"),

    ("test_movie_with_extras_structure",
     b"d4:infod4:name28:Movie.Name.2024.1080p.BluRay5:filesl"
     b"d4:pathl32:Movie.Name.2024.1080p.BluRay.mkve6:lengthi8000000000eee"
     b"d4:pathl6:Extras21:Behind.The.Scenes.mkve6:lengthi500000000eee"
     b"d4:pathl6:Extras11:Trailer.mkve6:lengthi100000000eeee"
     b"ee"),

    ("test_many_files_torrent (sample)",
     b"d4:infod4:name9:ManyFiles5:filesl"
     b"d4:pathl6:file00e6:lengthi100eee"
     b"d4:pathl6:file01e6:lengthi100eee"
     b"d4:pathl6:file49e6:lengthi100eeee"
     b"ee"),

    ("test_parse_multi_file_torrent",
     b"d4:infod4:name4:Test5:filesl"
     b"d4:pathl9:file1.txt9:file2.txte6:lengthi100eee"
     b"d4:pathl9:file3.txte6:lengthi200eeee"
     b"ee"),
]

print("Validating bencode test data...")
print("-" * 60)

all_valid = True
for name, data in tests:
    try:
        decoded = bdecode(data)
        if b'info' in decoded:
            info = decoded[b'info']
            if b'files' in info:
                print(
                    f"✓ {name}: Valid (multi-file, {len(info[b'files'])} files)")
            else:
                print(f"✓ {name}: Valid (single-file)")
        else:
            print(f"✓ {name}: Valid bencode")
    except Exception as e:
        print(f"✗ {name}: {e}")
        all_valid = False

print("-" * 60)
if all_valid:
    print("All bencode strings are valid!")
    sys.exit(0)
else:
    print("Some bencode strings have errors!")
    sys.exit(1)
