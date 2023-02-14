# 16-726 P1: Colorizing via Alignment

For this project we were given single-channel color slides taken in the early 19th century; the goal was to produce a full color composite, by aligning the individual pieces. Specifically, we would align red and green plates to the blue plate.

Minimal distortion was assumed, particularly in the centers of the image, so alignment is performed by sweeping for an optimal match in this region. This match is computed with your metric of choice, but we were prompted with NCC and SSD as simple starters. These were indeed sufficient in most cases.

The sweeping was done by searching a small grid; for larger images this can be sped up using coarse-to-fine multiresolution methods. Specifically, I resized all images such that the largest dimension was under 100 pixels, and assume that the offset is less than 4 pixels (i.e. 4%) in this resolution. After finding the optimal match at this small resolution I would resize back up by factors of 2; adjusting the search range accordingly.

Some issues: notably, SSD is pretty brittle and actually fails if the search range is too large; a small translation assumption is helpful. NCC is more robust; I'm sure other methods would be even better. I also suspect sweeping for more than I'm upsampling by at each step is unnecessary.

## Results
### Example set

Initial results shown with the simplest SSD metric.

![Cathedral](./out/numpy_base/ssd/cathedral_g_(5%2C%202)_r_(12%2C%203).jpg)

![Self Portrait](./out/numpy_base/ssd/self_portrait_g_(78%2C%2029)_r_(175%2C%2037).jpg)

![Icon](./out/numpy_base/ssd/icon_g_(41%2C%2017)_r_(90%2C%2023).jpg)


![Lady](./out/numpy_base/ssd/lady_g_(55%2C%208)_r_(115%2C%2012).jpg)

![Three generations](./out/numpy_base/ssd/three_generations_g_(52%2C%2014)_r_(111%2C%2012).jpg)

![Turkmen](./out/numpy_base/ssd/turkmen_g_(55%2C%2021)_r_(116%2C%2029).jpg)

![Village](./out/numpy_base/ssd/village_g_(64%2C%2012)_r_(137%2C%2022).jpg)

![Train](./out/numpy_base/ssd/train_g_(43%2C%206)_r_(86%2C%2032).jpg)

![Harvesters](./out/numpy_base/ssd/harvesters_g_(59%2C%2017)_r_(123%2C%2014).jpg)
There's an artifact on the left of this image but this seems unlikely to be a misalignment since the center of the image is fine. Rather, perhaps the person moved, or there's local warping.


![Emir](./out/numpy_base/ssd/emir_g_(48%2C%2024)_r_(136%2C%20-280).jpg)
This is a drastic misalignment.

| Image | Green Offset (X, Y) | Red Offset (X, Y) |
| ---: | --- | --- |
| Cathedral | 5, 2 | 12, 3 |
| Self-portrait | 78, 29 | 175, 37 |
| Icon | 41, 17 | 90, 23|
| Lady | 55, 8 | 115, 12 |
| Three-gen | 52, 14,| 111, 12 |
| Harvesters | 59, 17 | 123, 14 |
| Turkmen | 55, 21 | 116, 29 |
| Village | 64, 12 | 137, 22 |
| Train | 43, 6 | 86, 32 |
| Emir | 48, 24 | 136, -280 |

**Then the NCC metric.**

There are only 1 pixel disagreements except for Emir, so I only re-post Emir. It is still, however, a failure.
![Emir](./out/numpy_base/ncc/emir_g_(48%2C%2024)_r_(-330%2C%2017).jpg)

| Image | Green Offset (X, Y) | Red Offset (X, Y) |
| ---: | --- | --- |
| Cathedral | 5, 2 | 12, 3 |
| Self-portrait | 78, 29 | 175, 37 |
| Icon | 41, 17 | 90 23 |
| Lady | 54, 8 | 116, 12 |
| Three-gen | 52, 14 | 111, 12 |
| Harvesters | 59, 17 | 123, 14 |
| Turkmen | 55, 21 | 116, 28 |
| Village | 64, 12 | 137, 22 |
| Train | 42, 6 | 86, 32 |
| Emir | 48, 24 | -330, 17 |

## Failure modes
The Emir mode fails in its red channel for both SSD and NCC metric. Examining the grayscale source, the actual luminance in the red channel looks entirely different from the other two (there's no intensity in the red channel). Any matching metric must look at the edges to be successful here.

### Prokudin-Gorskii extra examples
I noted some SSD fragility in development; this is reiterated her.SSD fails on at least one plate here, but NCC works without tuning hyperparameters. It's hard to pinpoint an exact qualitative difference between these and the given demos (to explain the total failure here), so I would conclude that SSD is just too brittle to use in general.

**SSD**
![River](./out/numpy_base/ssd/river_g_(25%2C%20-5)_r_(153%2C%20-213).jpg)
![Railroad](./out/numpy_base/ssd/railroad_g_(101%2C%20-428)_r_(102%2C%202).jpg)
![Grass](./out/numpy_base/ssd/grass_g_(-484%2C%20-474)_r_(52%2C%2021).jpg)

**NCC**
![River](./out/numpy_base/ncc/river_g_(25%2C%20-5)_r_(102%2C%20-11).jpg)
![Railroad](./out/numpy_base/ncc/railroad_g_(26%2C%201)_r_(102%2C%202).jpg)
![Grass](./out/numpy_base/ncc/grass_g_(22%2C%2015)_r_(52%2C%2021).jpg)

| Image | SSD Green Offset | SSD Red Offset | NCC Green Offset | NCC Red Offset |
| ---: | --- | --- | --- | --- |
| River | 25, -5 | 153, -213 | 25, -5 | 102, -11 |
| Railroad | 101,-428 | 102, 2 | 26, 1 | 102, 2|
| Grass | -484, -474 | 52, 21 | 22, 15 | 52, 21 |




## Bells and Whistles - Pytorch
This produces no difference in computed offsets and so I won't repost images -- (posting two comptued offsets with NCC metric for brevity). I swapped all functions except the rescaling; which I still used `skt` for.

| Image | Green Offset (X, Y) | Red Offset (X, Y) |
| ---: | --- | --- |
| Self-portrait | 78, 29 | 175, 37 |
| Emir | 48, 24 | -330, 17 |