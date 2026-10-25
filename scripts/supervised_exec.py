"""[ENGINEERING] Wait for the bound GPU guard before exec; EOF refuses launch."""
import os
import sys
fd = int(sys.argv[1])
token = os.read(fd, 1)
os.close(fd)
if token != b'G':
    raise SystemExit(125)
os.execvp(sys.argv[2], sys.argv[2:])
