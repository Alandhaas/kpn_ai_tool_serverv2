import { useState, useEffect, useRef } from "react";
import { Settings, ChartColumn, ImageUp, LogOut } from "lucide-react";
import JSZip from "jszip";

import LoginScreen from "./components/LoginScreen";

import ImageUploadCard from "./components/ImageUploadCard";
import AdminPanel from "./admin/AdminPanel";
import "./App.css";

import DocsPdfViewer from "./components/DocsPdfViewer";

function App() {
   const ACCESS_CODE = "KPN2025";
   const [isLoggedIn, setIsLoggedIn] = useState(() => {
      return localStorage.getItem("kpn_access") === "true";
   });

   const abortRef = useRef(null);
   const [isHome, setIsHome] = useState(true);
   const [isAdmin, setIsAdmin] = useState(false);

   const [username, setUsername] = useState("");
   const [tempUsername, setTempUsername] = useState("");
   const [showUsernameModal, setShowUsernameModal] = useState(false);

   // File states (upload or camera)
   const [uploadedFile, setUploadedFile] = useState(null);

   // Results
   const [analysisResult, setAnalysisResult] = useState(null);
   const [isLoading, setIsLoading] = useState(false);
   const [error, setError] = useState(null);

   // Settings
   const [apiUrl, setApiUrl] = useState("http://localhost:8000");
   const [showSettings, setShowSettings] = useState(false);
   const [threshold, setThreshold] = useState(0.45);

   // GradCAM method (still selectable)
   const [selectedMethod, setSelectedMethod] = useState("gradcam++");
   const gradcamOptions = ["cam", "gradcam", "gradcam++"];

   useEffect(() => {
      const params = new URLSearchParams(window.location.search);
      const auth = params.get("auth");

      if (auth === ACCESS_CODE) {
         localStorage.setItem("kpn_access", "true");
         setIsLoggedIn(true);
      }
   }, []);

   // PROCESS IMAGE WHEN FILE IS UPLOADED OR SCANNED
   useEffect(() => {
      const fileToProcess = uploadedFile;
      if (!fileToProcess || !username) return;

      async function analyzeCBM() {
         setIsLoading(true);
         setError(null);
         setAnalysisResult(null);

         abortRef.current = new AbortController();

         try {
            // Prepare API call
            const formData = new FormData();
            formData.append("image", fileToProcess);
            formData.append("method", selectedMethod);
            formData.append("threshold", threshold);
            formData.append("username", username);

            const response = await fetch(`${apiUrl}/infer/gradcam/zip`, {
               method: "POST",
               body: formData,
               signal: abortRef.current.signal,
            });

            if (!response.ok) {
               const text = await response.text();
               console.error("API error:", text);
               throw new Error("Server returned an error.");
            }

            // The result is a ZIP file
            const zipBlob = await response.blob();
            const zip = await JSZip.loadAsync(zipBlob);

            let extracted = {
               images: {},
               data: null,
            };

            // Extract files from ZIP
            for (const name of Object.keys(zip.files)) {
               const file = zip.files[name];

               // JSON
               if (name.endsWith(".json")) {
                  const jsonText = await file.async("text");
                  extracted.data = JSON.parse(jsonText);
                  continue;
               }

               // Images
               if (name.endsWith(".jpg") || name.endsWith(".png")) {
                  const imgBlob = await file.async("blob");
                  extracted.images[name] = URL.createObjectURL(imgBlob);
               }
            }

            const meta = Array.isArray(extracted.data) ? extracted.data[0] : null;
            const imagesByConcept = {};
            if (meta && Array.isArray(meta.files)) {
               for (const entry of meta.files) {
                  imagesByConcept[entry.concept] = extracted.images[entry.filename];
               }
            }

            setAnalysisResult({
               meta,
               imagesByConcept,
            });
         } catch (err) {
            if (err.name === "AbortError") {
               console.log("API request cancelled.");
               return;
            }
            console.error("Error:", err);
            setError("An error occurred while analyzing the image.");
         } finally {
            setIsLoading(false);
            setUsername("");
            setTempUsername("");
            setShowUsernameModal(false);
         }
      }

      analyzeCBM();
   }, [uploadedFile, username]);

   return (
      <>
         {!isLoggedIn ? (
            <LoginScreen onLogin={() => setIsLoggedIn(true)} />
         ) : (
            <div className="flex flex-col min-h-screen bg-gradient-to-br from-green-500 to-emerald-600">
               {/* Navbar */}
               <nav className="bg-gradient-to-r from-green-50 to-green-100 shadow-sm border-b border-green-200">
                  <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                     <div className="flex justify-between items-center h-16 ps-3">
                        <img
                           src="/Logo_kpn.png"
                           alt="KPN Logo"
                           className="w-14 h-14 object-contain"
                        />

                        <div className="flex items-center space-x-3">
                           <button
                              onClick={() => setIsAdmin(!isAdmin)}
                              className="p-2 rounded-lg hover:bg-green-200 transition-colors">
                              <div onClick={() => setIsHome(!isHome)}>
                                 {isHome ? (
                                    <ChartColumn className="w-5 h-5 text-gray-700" />
                                 ) : (
                                    <ImageUp className="w-5 h-5 text-gray-700" />
                                 )}
                              </div>
                           </button>
                           <button
                              onClick={() => setShowSettings(true)}
                              className="p-2 rounded-lg hover:bg-green-200 transition-colors">
                              <Settings className="w-5 h-5 text-gray-700" />
                           </button>
                           <button
                              className="p-2 rounded-lg hover:bg-green-200 transition-colors"
                              onClick={() => {
                                 localStorage.removeItem("kpn_access");
                                 setIsLoggedIn(false);
                              }}>
                              <LogOut className="w-5 h-5 text-gray-700" />
                           </button>
                        </div>
                     </div>
                  </div>
               </nav>

               {isAdmin ? (
                  <div className="flex-grow">
                     <AdminPanel apiUrl={apiUrl} />
                  </div>
               ) : (
                  <>
                     <main className="flex-grow w-full px-2 sm:px-4 lg:px-8 py-6">
                        <div className="text-center mb-6">
                           <h2 className="text-2xl font-bold text-white mb-4">
                              FTU Installation Verification Using Computer Vision
                           </h2>
                           <p className="text-green-100 text-lg">
                              Upload an image or scan with your camera
                           </p>
                        </div>

                        <div className="flex flex-row items-center justify-center gap-6 w-full">
                           <div className="w-1/3 flex flex-col justify-center gap-6">
                              <ImageUploadCard
                                 imageFile={uploadedFile}
                                 setImageFile={(file) => {
                                    setUploadedFile(file);
                                    setShowUsernameModal(true);
                                 }}
                                 onClearImage={() => setShowUsernameModal(false)}
                                 onCancelImage={() => {
                                    setUploadedFile(null);
                                    setUsername("");
                                    setTempUsername("");
                                    setShowUsernameModal(false);
                                 }}
                              />
                              {isLoading && (
                                 <div className="flex flex-col items-center justify-center rounded-2xl p-4 space-y-4">
                                    <div className="text-white text-2xl font-medium animate-pulse">
                                       Analyzing image...
                                    </div>
                                    <button
                                       onClick={() => {
                                          if (abortRef.current) abortRef.current.abort();

                                          setUploadedFile(null);
                                          setUsername("");
                                          setTempUsername("");
                                          setShowUsernameModal(false);
                                          setAnalysisResult(null);
                                          setError(null);
                                          setIsLoading(false);
                                       }}
                                       className=" hover:opacity-90 text-white px-6 rounded-lg font-smal transition-all duration-200 hover:scale-105">
                                       Cancel
                                    </button>
                                 </div>
                              )}
                           </div>

                           <div className="flex-2 min-w-0">
                              <DocsPdfViewer />
                           </div>
                        </div>

                        {/* Usernsme */}
                        {showUsernameModal && (
                           <div className="fixed inset-0 flex items-center justify-center z-50">
                              <div className="absolute inset-0 bg-black/60 backdrop-blur-sm"></div>

                              <div className="relative bg-white/10 backdrop-blur-md text-white rounded-2xl shadow-2xl p-8 max-w-md w-[90%] z-10 animate-fade-in">
                                 <h3 className="text-2xl font-bold mb-6 text-center">
                                    Enter your name
                                 </h3>

                                 <input
                                    type="text"
                                    value={tempUsername}
                                    onChange={(e) => setTempUsername(e.target.value)}
                                    className="w-full bg-white/30 text-white rounded-lg px-4 py-2 mb-6 focus:outline-none focus:ring-2 focus:ring-emerald-300"
                                    placeholder="name"
                                 />

                                 <div className="flex justify-center">
                                    <button
                                       onClick={() => {
                                          if (!tempUsername.trim()) return;

                                          setUsername(tempUsername);
                                          setShowUsernameModal(false);
                                       }}
                                       className="bg-gradient-to-r from-green-500 to-emerald-600 hover:opacity-90 text-white px-6 py-2 rounded-lg font-medium shadow-md transition-all duration-200 hover:scale-105">
                                       Continue
                                    </button>
                                    <button
                                       onClick={() => {
                                          setUploadedFile(null);
                                          setUsername("");
                                          setTempUsername("");
                                          setShowUsernameModal(false);
                                       }}
                                       className=" hover:opacity-90 text-white px-6 py-2 rounded-lg font-medium transition-all duration-200 hover:scale-105">
                                       cancel
                                    </button>
                                 </div>
                              </div>
                           </div>
                        )}

                        {/* RESULT MODAL */}
                        {(analysisResult || error) && !isLoading && (
                           <div className="fixed inset-0 flex items-center justify-center z-50 ">
                              {/* BACKDROP */}
                              <div
                                 className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                                 onClick={() => {
                                    setAnalysisResult(null);
                                    setError(null);
                                 }}></div>

                              {/* MODAL */}
                              <div className="relative bg-black/80 backdrop-blur-md text-white rounded-2xl shadow-2xl p-8 w-full z-10 animate-fade-in max-h-[90vh] overflow-y-auto">
                                 {/* Title */}
                                 <h3 className="text-3xl font-bold mb-4 text-center">
                                    {error ? "Analysis Failed" : "Analysis Result"}
                                 </h3>

                                 {error ? (
                                    <div className="text-red-300 text-lg text-center">{error}</div>
                                 ) : (
                                    analysisResult?.meta && (
                                       <div className="text-center space-y-6 mx-auto">
                                          <p className="text-2xl font-bold">
                                             {analysisResult.meta.final_ok ? (
                                                <span className="text-green-400">
                                                   Correct Installation
                                                </span>
                                             ) : (
                                                <span className="text-red-400">
                                                   Incorrect Installation
                                                </span>
                                             )}
                                          </p>

                                          <div className="grid grid-cols-1 md:grid-cols-5 gap-5 mt-6">
                                             <div className="text-center">
                                                <p className="font-semibold mb-1">
                                                   Surrounding space
                                                </p>
                                                <p className="text-xs text-green-200 mb-2">
                                                   Score:{" "}
                                                   {analysisResult.meta.per_concept.rule_free_space.p_ok.toFixed(
                                                      3
                                                   )}
                                                </p>
                                                <img
                                                   src={
                                                      analysisResult.imagesByConcept
                                                         .rule_free_space
                                                   }
                                                   className="rounded-xl shadow-lg max-h-[800px] mx-auto object-contain"
                                                />
                                             </div>

                                             <div className="text-center">
                                                <p className="font-semibold mb-1">Cable Routing</p>
                                                <p className="text-xs text-green-200 mb-2">
                                                   Score:{" "}
                                                   {analysisResult.meta.per_concept.rule_cable_routing.p_ok.toFixed(
                                                      3
                                                   )}
                                                </p>
                                                <img
                                                   src={
                                                      analysisResult.imagesByConcept
                                                         .rule_cable_routing
                                                   }
                                                   className="rounded-xl shadow-lg max-h-[800px] mx-auto object-contain"
                                                />
                                             </div>
                                             <div className="text-center">
                                                <p className="font-semibold mb-1">Alignment</p>
                                                <p className="text-xs text-green-200 mb-2">
                                                   Score:{" "}
                                                   {analysisResult.meta.per_concept.rule_alignment.p_ok.toFixed(
                                                      3
                                                   )}
                                                </p>
                                                <img
                                                   src={
                                                      analysisResult.imagesByConcept.rule_alignment
                                                   }
                                                   className="rounded-xl shadow-lg max-h-[800px] mx-auto object-contain"
                                                />
                                             </div>
                                             <div className="text-center">
                                                <p className="font-semibold mb-1">FTU Covering</p>
                                                <p className="text-xs text-green-200 mb-2">
                                                   Score:{" "}
                                                   {analysisResult.meta.per_concept.rule_covering.p_ok.toFixed(
                                                      3
                                                   )}
                                                </p>
                                                <img
                                                   src={
                                                      analysisResult.imagesByConcept.rule_covering
                                                   }
                                                   className="rounded-xl shadow-lg max-h-[800px] mx-auto object-contain"
                                                />
                                             </div>
                                          </div>
                                       </div>
                                    )
                                 )}

                                 {/* Close button */}
                                 <div className="flex justify-center mt-6">
                                    <button
                                       onClick={() => {
                                          setUploadedFile(null);
                                          setAnalysisResult(null);
                                          setError(null);
                                          window.scrollTo({ top: 0, behavior: "smooth" });
                                       }}
                                       className="bg-gradient-to-r from-green-500 to-emerald-600 hover:opacity-90 text-white px-6 py-2 rounded-lg font-medium shadow-md transition-all hover:scale-105">
                                       Try Another
                                    </button>
                                 </div>
                              </div>
                           </div>
                        )}

                        {/* Loading */}
                     </main>
                  </>
               )}

               {/* SETTINGS MODAL */}
               {showSettings && (
                  <div className="fixed inset-0 flex items-center justify-center z-50">
                     <div
                        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                        onClick={() => setShowSettings(false)}></div>

                     <div className="relative bg-white/10 backdrop-blur-md text-white rounded-2xl shadow-2xl p-8 max-w-md w-[90%] z-10 animate-fade-in">
                        <h3 className="text-2xl font-bold mb-6 text-center">Settings</h3>

                        {/* API URL */}
                        <label className="block mb-2 text-sm font-semibold">Backend API URL:</label>
                        <input
                           type="text"
                           value={apiUrl}
                           onChange={(e) => setApiUrl(e.target.value)}
                           className="w-full bg-white/30 text-white rounded-lg px-4 py-2 mb-6"
                        />
                        <label className="block mb-2 text-sm font-semibold">Threshold</label>
                        <input
                           type="text"
                           value={threshold}
                           onChange={(e) => setThreshold(e.target.value)}
                           className="w-full bg-white/30 text-white rounded-lg px-4 py-2 mb-6"
                        />

                        <div className="flex justify-center">
                           <button
                              onClick={() => setShowSettings(false)}
                              className="bg-gradient-to-r from-green-500 to-emerald-600 hover:opacity-90 text-white px-6 py-2 rounded-lg font-medium">
                              Save
                           </button>
                        </div>
                     </div>
                  </div>
               )}

               {/* Footer */}
               <footer className="bg-gradient-to-r from-green-50 to-green-100 border-t border-green-200">
                  <div className="max-w-7xl mx-auto px-4 py-6 text-center text-gray-600 text-sm">
                     © {new Date().getFullYear()}{" "}
                     <span className="font-semibold text-green-600">AI Computer Vision</span> for
                     KPN & Inholland.
                  </div>
               </footer>
            </div>
         )}
      </>
   );
}

export default App;
