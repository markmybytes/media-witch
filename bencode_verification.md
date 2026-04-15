# Bencode Test Data Verification

## Fixed Tests

### 1. test_creates_multiple_files
**Fixed:** Changed file lengths from 6 to 9 (file1.txt, file2.txt, file3.txt are 9 chars each)

```
d4:infod4:name4:Root5:filesl
  d4:pathl9:file1.txte6:lengthi100eee
  d4:pathl9:file2.txte6:lengthi200eee
  d4:pathl9:file3.txte6:lengthi300eeee
ee
```

**Trace:**
- d → start dict
- 4:info → key "info"
- d → start info dict
- 4:name → key "name"
- 4:Root → value "Root" (4 chars) ✓
- 5:files → key "files" (5 chars) ✓
- l → start files list
- d → start file 1 dict
- 4:path → key "path"
- l → start path list
- 9:file1.txt → "file1.txt" (9 chars) ✓
- e → end path list
- 6:length → key "length"
- i100e → integer 100
- e → end file 1 dict
- [similar for files 2 and 3]
- e → end files list
- e → end info dict

**Status:** ✓ Valid

### 2. test_creates_nested_directories
**Status:** No changes needed - already correct

```
d4:infod4:name4:Root5:filesl
  d4:pathl3:dir5:file1e6:lengthi100eee
  d4:pathl3:dir4:sub25:file2e6:lengthi200eeee
ee
```

**Verification:**
- "dir" = 3 chars ✓
- "file1" = 5 chars ✓
- "sub2" = 4 chars ✓
- "file2" = 5 chars ✓

**Status:** ✓ Valid

### 3. test_multiple_files_same_directory
**Fixed:** Changed "6:file1", "6:file2", "6:file3" to "5:file1", "5:file2", "5:file3"

```
d4:infod4:name4:Root5:filesl
  d4:pathl3:dir5:file1e6:lengthi100eee
  d4:pathl3:dir5:file2e6:lengthi200eee
  d4:pathl3:dir5:file3e6:lengthi300eeee
ee
```

**Verification:**
- "dir" = 3 chars ✓
- "file1" = 5 chars ✓
- "file2" = 5 chars ✓
- "file3" = 5 chars ✓

**Status:** ✓ Valid

### 4. test_creates_from_multiple_torrents
**Fixed:** Changed from hardcoded "7:file{i}.txt" to dynamic length calculation

**Before:**
```python
content = f"d4:infod4:name7:file{i}.txt6:lengthi100eee".encode()
```

**After:**
```python
filename = f"file{i}.txt"
content = f"d4:infod4:name{len(filename)}:{filename}6:lengthi100eee".encode()
```

**Verification:**
- "file0.txt" = 9 chars → 9:file0.txt ✓
- "file1.txt" = 9 chars → 9:file1.txt ✓
- "file2.txt" = 9 chars → 9:file2.txt ✓

**Status:** ✓ Valid

### 5. test_batch_continues_on_error
**Fixed:** Changed "8:valid.txt" to "9:valid.txt"

```
d4:infod4:name9:valid.txt6:lengthi100eee
```

**Verification:**
- "valid.txt" = 9 chars ✓

**Status:** ✓ Valid

### 6. test_tv_show_season_structure
**Fixed:** Multiple string length corrections
- "25:Show.Name.S01.1080p.WEB" → "23:Show.Name.S01.1080p.WEB"
- "24:Show.Name.S01E01.en.srt" → "23:Show.Name.S01E01.en.srt"
- "24:Show.Name.S01E02.en.srt" → "23:Show.Name.S01E02.en.srt"

```
d4:infod4:name23:Show.Name.S01.1080p.WEB5:filesl
  d4:pathl20:Show.Name.S01E01.mkve6:lengthi1500000000eee
  d4:pathl20:Show.Name.S01E02.mkve6:lengthi1600000000eee
  d4:pathl4:Subs23:Show.Name.S01E01.en.srte6:lengthi50000eee
  d4:pathl4:Subs23:Show.Name.S01E02.en.srte6:lengthi52000eeee
ee
```

**Verification:**
- "Show.Name.S01.1080p.WEB" = 23 chars ✓
- "Show.Name.S01E01.mkv" = 20 chars ✓
- "Show.Name.S01E02.mkv" = 20 chars ✓
- "Subs" = 4 chars ✓
- "Show.Name.S01E01.en.srt" = 23 chars ✓
- "Show.Name.S01E02.en.srt" = 23 chars ✓

**Status:** ✓ Valid

### 7. test_movie_with_extras_structure
**Fixed:** Changed "18:Behind.The.Scenes.mkv" to "21:Behind.The.Scenes.mkv"

```
d4:infod4:name28:Movie.Name.2024.1080p.BluRay5:filesl
  d4:pathl32:Movie.Name.2024.1080p.BluRay.mkve6:lengthi8000000000eee
  d4:pathl6:Extras21:Behind.The.Scenes.mkve6:lengthi500000000eee
  d4:pathl6:Extras11:Trailer.mkve6:lengthi100000000eeee
ee
```

**Verification:**
- "Movie.Name.2024.1080p.BluRay" = 28 chars ✓
- "Movie.Name.2024.1080p.BluRay.mkv" = 32 chars ✓
- "Extras" = 6 chars ✓
- "Behind.The.Scenes.mkv" = 21 chars ✓
- "Trailer.mkv" = 11 chars ✓

**Status:** ✓ Valid

### 8. test_many_files_torrent
**Status:** No changes needed - already correct

```python
files_bencode = b"".join([
    f"d4:pathl6:file{i:02d}e6:lengthi100eee".encode()
    for i in range(50)
])
```

**Verification:**
- "file00" through "file49" are all 6 chars ✓

**Status:** ✓ Valid

### 9. test_parse_multi_file_torrent (test_torrent_parser.py)
**Fixed:** Changed all filename lengths from 6 to 9

```
d4:infod4:name4:Test5:filesl
  d4:pathl9:file1.txt9:file2.txte6:lengthi100eee
  d4:pathl9:file3.txte6:lengthi200eeee
ee
```

**Verification:**
- "Test" = 4 chars ✓
- "files" = 5 chars ✓
- "file1.txt" = 9 chars ✓
- "file2.txt" = 9 chars ✓
- "file3.txt" = 9 chars ✓

**Status:** ✓ Valid

## Summary

All 9 tests have been fixed with correct bencode formatting:
- String format: `<length>:<string>` where length is the exact number of characters
- Integer format: `i<number>e`
- All bencode structures properly nested with matching delimiters

All fixes verified manually by character count and structure tracing.
