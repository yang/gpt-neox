# Notes

## NeoX Container Building

Doesn't seem to like python 3.10 from recent nvcr.io images

23.04 is last image with 3.8 (no 3.9 images for some reason)

So try to bump cuda base image to 12.2 and run the dockerfile as is

### Dependencies

Bump apex to 23.08 commit...previous was from 2021
Remove requirements-sparseattention pin of old triton that forces torch 1.x vs 2.x