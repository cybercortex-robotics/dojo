var map;
var polyline_m = null;
var global_index = 0;

function update_path(map)
{
	if (polyline_m != null)
	{
		map.removeLayer(polyline_m);
	}

	local_markers = Array();
	map.eachLayer(function(layer) {
		if (layer instanceof L.Marker) {
			local_markers.push(layer.getLatLng());
		}
	});

	polyline_m = L.polyline(local_markers, {color: 'red'});
	map.addLayer(polyline_m);
}

function initialize()
{
    map = L.map('map').setView([45.670339, 25.549457], 17);

    L.tileLayer('https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png',
	{
		attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
		maxZoom: 20
	}).addTo(map);

	new QWebChannel(qt.webChannelTransport, function (channel)
	{
		window.MapWindow = channel.objects.MapWindow;

		if(typeof MapWindow != 'undefined')
		{
			map.on('click', function (e)
			{
				var point = e.latlng;
				let marker = L.marker(point, { riseOnHover: true, draggable: true });
				const idx = global_index;

				marker.on("click", (e) => delete_marker(idx, marker));
				marker.on("drag", (e) => drag_marker(idx, marker));

				map.addLayer(marker);
				MapWindow.on_add_marker(idx, point.lat, point.lng);

				global_index = global_index + 1;

				update_path(map);
			});
		}
	});
}

function argmin(vec) {
	let minIdx = 0;
	for (let i = 0; i < vec.length; i++) {
		if (vec[i] < vec[minIdx]) {
			minIdx = i;
		}
	}
	
	return minIdx;
}

function delete_marker(idx, marker)
{
	map.removeLayer(marker);
	MapWindow.on_remove_marker(idx);

	update_path(map);
}

function drag_marker(idx, marker)
{
	MapWindow.on_update_marker(
		idx,
		marker.getLatLng().lat,
		marker.getLatLng().lng);

	update_path(map);
}

function clear_markers() {
	// TODO: better clearing method
	local_markers = Array();
	map.eachLayer(function(layer) {
		if (layer instanceof L.Marker) {
			local_markers.push(layer);
		}
	});
	local_markers.forEach(function (layer) {
		map.removeLayer(layer);
	});
}

function add_marker_at(lat, lng) {
	let point = L.latLng(lat, lng);
	let marker = L.marker(point, { riseOnHover: true, draggable: true });
	const idx = global_index;

	marker.on("click", (e) => delete_marker(idx, marker));
	marker.on("drag", (e) => drag_marker(idx, marker));

	map.addLayer(marker);
	MapWindow.on_add_marker(idx, point.lat, point.lng);

	global_index = global_index + 1;

	update_path(map);
}

function delete_marker_at(lat, lng) {
	local_points = Array();
	local_markers = Array();
	map.eachLayer(function(layer) {
		if (layer instanceof L.Marker) {
			local_points.push(layer.getLatLng());
			local_markers.push(layer);
		}
	});
	
	distances = local_points.map(
		(m) => Math.sqrt((m.lat - lat)*(m.lat - lat) + (m.lng - lng)*(m.lng - lng))
	);
	
	const minIdx = argmin(distances);
	
	map.removeLayer(local_markers[minIdx]);
	update_path(map);
}

function move_marker_at(lat, lng, new_lat, new_lng) {
	local_points = Array();
	local_markers = Array();
	map.eachLayer(function(layer) {
		if (layer instanceof L.Marker) {
			local_points.push(layer.getLatLng());
			local_markers.push(layer);
		}
	});
	
	distances = local_points.map(
		(m) => Math.sqrt((m.lat - lat)*(m.lat - lat) + (m.lng - lng)*(m.lng - lng))
	);
	
	const minIdx = argmin(distances);
	
	local_markers[minIdx].setLatLng(L.latLng(new_lat, new_lng));
	update_path(map);
}
