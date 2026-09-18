import { useEffect, useState } from "react"

export default function App() {

  const [data, setData] = useState(null)

  useEffect(() => {

    fetch("http://127.0.0.1:8000/predict")
      .then(res => res.json())
      .then(data => setData(data))

  }, [])

  return (
    <div className="min-h-screen p-10 bg-gray-100">

      <h1 className="text-4xl font-bold mb-6">
        EV Guidance System
      </h1>

      {data && (
        <div className="bg-white rounded-2xl shadow p-6">

          <h2 className="text-2xl font-semibold">
            Estimated Range
          </h2>

          <p className="text-5xl mt-4">
            {data.estimated_range_km} km
          </p>

          <p className="mt-4">
            Confidence: {data.confidence}
          </p>

          <p className="mt-2 text-gray-600">
            {data.explanation}
          </p>

        </div>
      )}

    </div>
  )
}