import React, { useMemo, useState } from "react";
import { CheckCircle2, MapPin, Ruler, Search, X } from "lucide-react";
import { DISEASE_STYLES, type Detection } from "../types";
import { imageUrl } from "../api";

interface Props {
  detections: Detection[];
  initialSelected: Detection | null;
  onClearInitialSelected: () => void;
  onUpdateDetection: (id: number, reviewed: boolean, note?: string) => Promise<void>;
}

export const DetectionsTab: React.FC<Props> = ({
  detections,
  initialSelected,
  onClearInitialSelected,
  onUpdateDetection,
}) => {
  const [clsFilter, setClsFilter] = useState<string>("diseased");
  const [rowFilter, setRowFilter] = useState<string>("all");
  const [search, setSearch] = useState("");
  const [modalDet, setModalDet] = useState<Detection | null>(initialSelected);
  const [noteDraft, setNoteDraft] = useState(initialSelected?.operator_note ?? "");

  React.useEffect(() => {
    if (initialSelected) {
      setModalDet(initialSelected);
      setNoteDraft(initialSelected.operator_note ?? "");
    }
  }, [initialSelected]);

  const filtered = useMemo(() => {
    return detections.filter((d) => {
      if (clsFilter === "diseased" && d.disease_class === "healthy") return false;
      if (clsFilter !== "all" && clsFilter !== "diseased" && d.disease_class !== clsFilter) {
        return false;
      }
      if (rowFilter !== "all" && String(d.row_index) !== rowFilter) return false;
      if (search.trim()) {
        const q = search.toLowerCase();
        const hay = `${d.disease_class} row ${d.row_index} ${d.lane} ${d.operator_note}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [detections, clsFilter, rowFilter, search]);

  const openModal = (d: Detection) => {
    setModalDet(d);
    setNoteDraft(d.operator_note ?? "");
  };

  const closeModal = () => {
    setModalDet(null);
    onClearInitialSelected();
  };

  return (
    <div className="space-y-5">
      {/* Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200/90 bg-white px-4 py-3">
        <div className="flex flex-wrap items-center gap-1.5">
          {[
            { k: "diseased", label: "Diseased Only" },
            { k: "anthracnose_fruit_rot", label: "Anthracnose" },
            { k: "gray_mold", label: "Gray Mold" },
            { k: "powdery_mildew", label: "Powdery Mildew" },
            { k: "all", label: "All Captures" },
          ].map((item) => (
            <button
              key={item.k}
              onClick={() => setClsFilter(item.k)}
              className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition ${
                clsFilter === item.k
                  ? "bg-slate-900 text-white"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <select
            value={rowFilter}
            onChange={(e) => setRowFilter(e.target.value)}
            className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-700"
          >
            <option value="all">All Rows</option>
            <option value="1">Row 1</option>
            <option value="2">Row 2</option>
            <option value="3">Row 3</option>
            <option value="4">Row 4</option>
          </select>

          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-slate-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter captures…"
              className="w-44 rounded-lg border border-slate-200 bg-slate-50 py-1.5 pl-8 pr-3 text-xs text-slate-800 focus:border-emerald-500 focus:bg-white focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* Captures Grid */}
      {filtered.length === 0 ? (
        <div className="rounded-2xl border border-slate-200 bg-white p-12 text-center text-sm text-slate-500">
          {detections.length === 0
            ? "0 captures recorded yet — real camera footage will appear here automatically as YOLOv11 detects strawberries during your Gazebo trial."
            : "No captures match the selected filters."}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((det) => {
            const st = DISEASE_STYLES[det.disease_class] ?? DISEASE_STYLES.healthy;
            return (
              <div
                key={det.id}
                onClick={() => openModal(det)}
                className="group cursor-pointer overflow-hidden rounded-2xl border border-slate-200/90 bg-white transition hover:border-slate-300 hover:shadow-sm"
              >
                <div className="relative aspect-[4/3] overflow-hidden bg-slate-100">
                  <img
                    src={imageUrl(det.image_path)}
                    alt={st.short}
                    className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.03]"
                    loading="lazy"
                  />
                  <span
                    className={`absolute left-2.5 top-2.5 rounded-md border px-2 py-0.5 text-[11px] font-semibold shadow-2xs ${st.bg} ${st.text} ${st.border}`}
                  >
                    {st.short}
                  </span>
                  <span className="absolute right-2.5 top-2.5 rounded-md bg-slate-900/75 px-1.5 py-0.5 font-mono text-[11px] text-white">
                    {Math.round(det.confidence * 100)}%
                  </span>
                  {det.reviewed && (
                    <span className="absolute bottom-2 right-2 inline-flex items-center gap-1 rounded-md bg-emerald-600/90 px-2 py-0.5 text-[10px] font-medium text-white">
                      <CheckCircle2 className="h-3 w-3" /> Reviewed
                    </span>
                  )}
                </div>

                <div className="p-3.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-900">
                      Row {det.row_index}{" "}
                      <span className="font-normal text-slate-400">({det.lane})</span>
                    </span>
                    <span className="font-mono text-[11px] text-slate-500">
                      ({det.x_m.toFixed(2)}, {det.y_m.toFixed(2)}m)
                    </span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between text-[11px] text-slate-400">
                    <span>
                      Mast {det.mast_height_m.toFixed(2)}m · {det.camera_side} cam
                    </span>
                    <span>
                      {new Date(det.timestamp).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Detail Inspection Modal */}
      {modalDet && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4 backdrop-blur-xs"
          onClick={closeModal}
        >
          <div
            className="w-full max-w-2xl overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3.5">
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-slate-400">#{modalDet.id}</span>
                <h3 className="text-sm font-semibold text-slate-900">
                  {(DISEASE_STYLES[modalDet.disease_class] ?? DISEASE_STYLES.healthy).label}
                </h3>
              </div>
              <button
                onClick={closeModal}
                className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="grid grid-cols-1 gap-5 p-5 md:grid-cols-2">
              <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
                <img
                  src={imageUrl(modalDet.image_path)}
                  alt={modalDet.disease_class}
                  className="aspect-square w-full object-cover"
                />
              </div>

              <div className="flex flex-col justify-between space-y-4">
                <div className="space-y-2.5 text-xs">
                  <div className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2">
                    <span className="inline-flex items-center gap-1.5 text-slate-500">
                      <MapPin className="h-3.5 w-3.5" /> Coordinates
                    </span>
                    <span className="font-mono font-semibold text-slate-900">
                      X {modalDet.x_m.toFixed(2)}m · Y {modalDet.y_m.toFixed(2)}m
                    </span>
                  </div>
                  <div className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2">
                    <span className="text-slate-500">Row / Lane</span>
                    <span className="font-medium text-slate-900">
                      Row {modalDet.row_index} · {modalDet.lane} ({modalDet.camera_side} cam)
                    </span>
                  </div>
                  <div className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2">
                    <span className="inline-flex items-center gap-1.5 text-slate-500">
                      <Ruler className="h-3.5 w-3.5" /> Mast Optical Height
                    </span>
                    <span className="font-mono font-semibold text-slate-900">
                      {modalDet.mast_height_m.toFixed(3)} m
                    </span>
                  </div>
                  <div className="flex items-center justify-between rounded-xl bg-slate-50 px-3 py-2">
                    <span className="text-slate-500">YOLOv11 Confidence</span>
                    <span className="font-mono font-semibold text-slate-900">
                      {(modalDet.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div>
                    <label className="mb-1 block text-[11px] font-medium text-slate-500">
                      Operator Treatment Note (saved to SQLite)
                    </label>
                    <textarea
                      rows={3}
                      value={noteDraft}
                      onChange={(e) => setNoteDraft(e.target.value)}
                      placeholder="e.g., Flagged for localized copper fungicide treatment…"
                      className="w-full rounded-xl border border-slate-200 p-2.5 text-xs text-slate-800 focus:border-emerald-500 focus:outline-none"
                    />
                  </div>
                </div>

                <div className="flex items-center justify-end gap-2">
                  <button
                    onClick={async () => {
                      const nextRev = !modalDet.reviewed;
                      await onUpdateDetection(modalDet.id, nextRev, noteDraft);
                      setModalDet({ ...modalDet, reviewed: nextRev, operator_note: noteDraft });
                    }}
                    className={`inline-flex items-center gap-1.5 rounded-xl px-3.5 py-2 text-xs font-semibold transition ${
                      modalDet.reviewed
                        ? "border border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "bg-slate-900 text-white hover:bg-slate-800"
                    }`}
                  >
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    {modalDet.reviewed ? "Saved & Reviewed" : "Save & Mark Reviewed"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
