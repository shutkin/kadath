use log::info;
use crate::orderbook_values::OrderBookValues;
use model::common::EmptyResult;
use plotters::prelude::*;

pub fn draw_chart(values: &[OrderBookValues], horizons: &[f64], predictions: &[f64], filename: &str) -> EmptyResult {
    let root = BitMapBackend::new(filename, (8000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let h_min = horizons.iter().copied().reduce(f64::min).unwrap_or_default();
    let h_max = horizons.iter().copied().reduce(f64::max).unwrap_or_default();
    let p_min = predictions.iter().copied().reduce(f64::min).unwrap_or_default();
    let p_max = predictions.iter().copied().reduce(f64::max).unwrap_or_default();
    info!("Horizons: {h_min} - {h_max}, Predictions: {p_min} - {p_max}");

    let start_timestamp = values[0].time.timestamp_millis() as f64;
    let x_data_minutes = values
        .iter()
        .map(|v| {
            (v.time.timestamp_millis() as f64 - start_timestamp) / 1000.0 / 60.0
        })
        .collect::<Vec<_>>();

    let mut chart = ChartBuilder::on(&root)
        .caption("Prediction vs Horizon", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..x_data_minutes[values.len() - 1], h_min.min(p_min)..h_max.max(p_max))?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            x_data_minutes.iter().copied().zip(predictions.iter().copied()),
            RED.mix(0.5),
        ))?
        .label("Predictions")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.5)));

    chart
        .draw_series(LineSeries::new(
            x_data_minutes.iter().copied().zip(horizons.iter().copied()),
            BLUE.mix(0.5),
        ))?
        .label("Horizons")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.mix(0.5)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.66))
        .border_style(full_palette::GREY)
        .draw()?;

    // Save the buffer to disk
    root.present()?;
    Ok(())
}