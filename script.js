let model;

let valoresPerdida = [];

async function entrenarModelo() {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  const xs = tf.tensor2d([-9 ,-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [21, 1]);
  const ys = tf.tensor2d([-12 ,-10, -8 ,-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28], [21, 1]);

  valoresPerdida = [];

  await model.fit(xs, ys, {
    epochs: 380,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        valoresPerdida.push({ epoch: epoch, loss: logs.loss });
        console.log(`Época ${epoch +1}: pérdida = ${logs.loss.toFixed(4)}`);
      },
      onTrainEnd: () => {
        document.getElementById("estadoEntrenamiento").innerHTML =
          "Entrenamiento terminado, ya puede predecir";
        graficarPerdida();
      },
    },
  });
}

function predecirValor() {
  const inputStr = document.getElementById("inputX").value;
  if (!inputStr.trim()) {
    document.getElementById("resultado").innerHTML =
      "Por favor ingrese valores para predecir";
    return;
  }
  const valores = inputStr
    .split(",")
    .map((v) => parseFloat(v.trim()))
    .filter((v) => !isNaN(v));

  if (valores.length === 0) {
    document.getElementById("resultado").innerHTML =
      "Entrada no válida. Ingrese números separados por comas.";
    return;
  }

  const prediccion = model.predict(tf.tensor2d(valores, [valores.length, 1]));
  const resultados = prediccion.dataSync();

  // mostrar el resultado en una tabla
  let html =
  "<h4>Resultados de predicción:</h4><table cellpadding='5'><tr><th>x</th><th>predicho</th></tr>";

valores.forEach((x, i) => {
  html += `<tr><td>${x}</td><td>${resultados[i].toFixed(2)}</td></tr>`;
});

  html += "</table>";
  document.getElementById("resultado").innerHTML = html;
}

function graficarPerdida() {
  const trace = {
    x: valoresPerdida.map((d) => d.epoch), // eje x : epocas
    y: valoresPerdida.map((d) => d.loss), // eje y: valores de pérdida
    mode: "lines", // tipo de grafico
    name: "Pérdida", // nombre de la serie
    line: { color: "purple", width: 2 },
  };
  // config del siseño del grafico
  const layout = {
    title: "Pérdida durante el entrenamiento",
    xaxis: { title: "Época" },
    yaxis: { title: "Pérdida" },
  };
  // renderizar con plotly
  Plotly.newPlot("graficoPerdida", [trace], layout);
}

// Iniciar el entrenamiento al cargar la página
document.addEventListener("DOMContentLoaded", entrenarModelo);