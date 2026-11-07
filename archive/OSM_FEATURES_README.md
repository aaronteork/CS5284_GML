# OSM features: data source and preparation

Work in progress. This file will grow into the data dictionary (columns, groups, fill rates, known limits). For now it records where the OpenStreetMap data comes from and how it was prepared.

## Where the OSM data comes from

We use a Geofabrik state extract, not a live download from the Overpass API.

- **Why:** on 2026-10-20 the public Overpass server (`overpass-api.de`) answered `502 Bad Gateway: server is currently at capacity`, so the download in the notebook failed. A Geofabrik file is a fixed snapshot, it does not time out, and every run uses the same OSM data.
- **Source page:** https://download.geofabrik.de/north-america/us.html (pick the state from the list).
- **File used for Delaware:** `delaware-261018.osm.pbf`, saved in `MyDrive/cs5284_project/data/`.
- **Snapshot date:** 2026-10-18, read from the file name (yymmdd). Confirm against the date shown on the Geofabrik page.
- OSM is a snapshot of today's roads, while the accident labels span about 2009 to 2022 (see the limitations in the brief).

## Steps to reproduce (Delaware; repeat with the Maryland file for MD)

1. Open https://download.geofabrik.de/north-america/us.html, click **Delaware**, and download the `.osm.pbf` file.
2. Upload it to `MyDrive/cs5284_project/data/`. Do not unzip or rename it.
3. In Colab, run `osm_pipeline.ipynb` (Run All). It calls `prepare_osm_roads_zip` and `prepare_osm_places_zip` itself for any state whose features are not already saved, and loads the saved `{STATE}_osm_edge_features.parquet` from Drive otherwise. For Maryland, upload `maryland-YYMMDD.osm.pbf` first. The preparation steps skip the work if the zip is already on Drive. (`osm_eda.ipynb` is the earlier working log.) `prepare_osm_roads_zip` does the following:
   1. Installs `osmium-tool` (`apt-get`) if it is not already available.
   2. **Filters** the `.pbf` with `osmium tags-filter`, keeping only ways tagged with one of 14 `highway` values: motorway, motorway_link, trunk, trunk_link, primary, primary_link, secondary, secondary_link, tertiary, tertiary_link, unclassified, residential, living_street, road. The nodes those ways use are kept with their tags.
   3. **Converts** the result to `.osm` XML with `osmium cat`. This is the format osmnx can read (`graph_from_xml`); osmnx cannot read `.pbf` directly.
   4. **Zips** the XML as `DE_osm_roads.zip` in `MyDrive/cs5284_project/data/`.
4. The notebook then builds the graph from that zip with `load_osm_drive_network_from_zip`.

Output of the Delaware run (done by hand before the method existed): filtered `.pbf` 4.8 MB, XML 71 MB, zip 9.3 MB. Only the zip is kept on Drive; the other two live in Colab's temporary storage.

`prepare_osm_places_zip` makes `{STATE}_osm_places.zip` the same way, from the full `.pbf`: nodes, ways and relations tagged `amenity` = school, bar, pub, restaurant, fast_food or parking, or `highway=bus_stop`, exported as GeoJSON (areas become polygons; their centre is used). These feed the g4 "surroundings" features.

## Why these 14 road types

They are the highway classes the benchmark dataset already has (`living_street` through `unclassified` in its `edges` keys). The list also matches what osmnx's `drive` network includes, and leaves out `service`, `footway`, `cycleway`, `track` and similar.

## Known differences from an osmnx `drive` download

- osmnx's drive network also drops ways tagged `motor_vehicle=no`, `motorcar=no`, `area=yes`, and some private or alley service roads. Our filter does not, so OSM counts may be slightly higher. This is checked in the node and road size comparison (dataset DE: 49,023 nodes, 116,196 directed roads).
- The extract covers the whole state. The 1 km buffered study area used for an Overpass download is not needed here.
- The original `.pbf` is untouched on Drive, so points of interest for the surroundings group (g4) can be taken from it later.
- Note: `utils.read_data` unpacks every zip in `data/`, so it also unpacks `DE_osm_roads.zip` into `/content/data/`.
