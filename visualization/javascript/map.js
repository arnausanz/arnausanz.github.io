var map = L.map('map', {
    center: [41.75, 1.7],
    zoom: 8,
    zoomControl: false,
    dragging: false,
    scrollWheelZoom: false,
    doubleClickZoom: false
});

L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png', {
    attribution: '&copy; Arnau Sanz'
}).addTo(map);

const GeoJson = 'visualization/geojson/catalonia.geo.json';

fetch(GeoJson)
    .then(function(response) {
        return response.json();
    })
    .then(function(data) {
        var style = {
            "color": "black",
            "weight": 1,
            "opacity": 0.5,
            "fillOpacity": 0
        };

        L.geoJSON(data, {
            style: style,
            interactive: false
        }).addTo(map);
    });

var sensorLayer = L.layerGroup();

var embassamentIcon = L.divIcon({
    className: 'embassament-icon',
    html: `
        <svg width="25" height="25" viewBox="-2.5 -2.5 25.5 25.5">
            <circle cx="12" cy="12" r="10" fill="#6BB8FF" stroke="#329DFF" stroke-width="1.5"/>
            <path d="M6,13 Q12,9 18,13 Q12,17 6,13 Z" fill="#BCDFFF" opacity="1"/>
            <path d="M6,17 Q12,13 18,17 Q12,21 6,17 Z" fill="#BCDFFF" opacity="1"/>
        </svg>
    `,
    iconSize: [30, 30],
    iconAnchor: [15, 15]
});


Promise.all([
    fetch("data/processed/aca/sensor_metadata_web.csv").then(response => response.text()),
    fetch("web_source.csv").then(response => response.text()),
    fetch("web_today.csv").then(response => response.text())
]).then(([sensorCsvText, webSourceCsvText, webTodayCsvText]) => {
    const parseCsv = (csvText) => {
        const lines = csvText.split("\n").filter(line => line.trim() !== "");
        const headers = lines[0].split(",");
        const values = lines[1].split(",");
        const result = {};
        headers.forEach((header, index) => {
            result[header.trim()] = values[index].trim();
        });
        return result;
    };

    const parseSensorCsv = (csvText) => {
        const lines = csvText.split("\n").filter(line => line.trim() !== "");
        const headers = lines[0].split(",");
        return lines.slice(1).map(line => {
            const values = line.split(",");
            const result = {};
            headers.forEach((header, index) => {
                result[header.trim()] = values[index].trim();
            });
            return result;
        });
    };

    const sensorData = parseSensorCsv(sensorCsvText);
    const webSourceData = parseCsv(webSourceCsvText);
    const webTodayData = parseCsv(webTodayCsvText);

    const webSourceMap = webSourceData;
    const webTodayMap = webTodayData;

    sensorData.forEach(point => {
        if (!isNaN(point.latitude) && !isNaN(point.longitude)) {
            const todayVolume = webTodayMap[point.name] || "Unknown volume";
            const expectedVolume = webSourceMap[point.name] || "Unknown volume";
            const popupContent = `<b>${point.name}</b><br><i>Today's volume:</i> <b>${todayVolume}</b><br><i>Volume expected (in 30 days):</i> <b>${expectedVolume}</b>`;
            L.marker([parseFloat(point.latitude), parseFloat(point.longitude)], {icon: embassamentIcon})
                .addTo(sensorLayer)
                .bindPopup(popupContent);
        }
    });

    sensorLayer.addTo(map);
});

var baseLayers = {};
var overlays = {
    "Embassament": sensorLayer
};

L.control.layers(baseLayers, overlays).addTo(map);