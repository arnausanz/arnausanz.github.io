var map = L.map('map', {
    center: [41.75, 1.7],   // Centra el mapa en coordenadas específicas
    zoom: 8,              // Nivel de zoom inicial
    zoomControl: false,   // Desactiva el control de zoom
    dragging: false,      // Deshabilita el desplazamiento del mapa
    scrollWheelZoom: false,  // Deshabilita el zoom con la rueda del ratón
    doubleClickZoom: false   // Deshabilita el zoom al hacer doble clic
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
        // Define el estilo para la capa GeoJSON
        var style = {
            "color": "black",       // Color de la línea del borde
            "weight": 1,          // Grosor de la línea del borde
            "opacity": 0.5,         // Opacidad de la línea del borde
            "fillOpacity": 0      // Opacidad de relleno del polígono (0 para eliminar el relleno)
        };

        // Añade la capa GeoJSON al mapa y desactiva la interactividad
        L.geoJSON(data, {
            style: style,
            interactive: false  // Desactiva la interactividad de la capa GeoJSON
        }).addTo(map);
    })

var sensorLayer = L.layerGroup();

var embassamentIcon = L.divIcon({
    className: 'embassament-icon',  // Definim la classe CSS
    html: `
        <svg width="25" height="25" viewBox="-2.5 -2.5 25.5 25.5">
            <!-- Cercle blau per a l'aigua -->
            <circle cx="12" cy="12" r="10" fill="#6BB8FF" stroke="#329DFF" stroke-width="1.5"/>
            <!-- Ones blanques -->
            <path d="M6,13 Q12,9 18,13 Q12,17 6,13 Z" fill="#BCDFFF" opacity="1"/>
            <path d="M6,17 Q12,13 18,17 Q12,21 6,17 Z" fill="#BCDFFF" opacity="1"/>
        </svg>
    `,
    iconSize: [30, 30],
    iconAnchor: [15, 15]
});

fetch("data/processed/aca/sensor_metadata.csv")
    .then(response => response.text())
    .then(csvText => {
        const lines = csvText.split("\n").filter(line => line.trim() !== "");
        const headers = lines[0].split(",");
        const points = lines.slice(1).map(line => {
            const values = line.split(",");
            const point = {};
            headers.forEach((header, index) => {
                point[header.trim()] = values[index].trim();
            });
            return {
                name: point.name,
                lat: parseFloat(point.latitude),
                lon: parseFloat(point.longitude)
            };
        });
        console.log(points);

        points.forEach(point => {
            if (!isNaN(point.lat) && !isNaN(point.lon)) {
                L.marker([point.lat, point.lon], {icon: embassamentIcon}).addTo(sensorLayer).bindPopup(point.name);
            }
        });

        sensorLayer.addTo(map);
    });

var baseLayers = {};
var overlays = {
    "Embassament": sensorLayer
};

L.control.layers(baseLayers, overlays).addTo(map);