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

const GeoJson = 'catalonia.geo.json';

fetch(GeoJson)
    .then(function(response) {
        return response.json();
    })
    .then(function(data) {
        // Define el estilo para la capa GeoJSON
        var style = {
            "color": "black",       // Color de la línea del borde
            "weight": 1.5,          // Grosor de la línea del borde
            "opacity": 0.25,         // Opacidad de la línea del borde
            "fillOpacity": 0      // Opacidad de relleno del polígono (0 para eliminar el relleno)
        };

        // Añade la capa GeoJSON al mapa y desactiva la interactividad
        L.geoJSON(data, {
            style: style,
            interactive: false  // Desactiva la interactividad de la capa GeoJSON
        }).addTo(map);
    })