// import React from "react";
// import { Link, useLocation } from "react-router-dom";
// // import { createPageUrl } from "../utils";

// const navigationItems = [
// ];

// export default function Layout({ children, currentPageName }) {
//   const location = useLocation();

//   return (
//     <div className="min-h-screen bg-white">
//       <nav className="bg-[#1a1a1a] border-b border-gray-800">
//         <div className="max-w-7xl mx-auto px-6">
//           <div className="flex items-center justify-between h-14">
//             <div className="flex items-center gap-2">
//               <div className="w-5 h-5 bg-white/10 rounded" />
//               <span className="text-white text-sm font-medium">
//                 quantiscribe-ai-scribe.lovable.app
//               </span>
//             </div>
//             <div className="hidden lg:flex items-center gap-1">
//               {navigationItems.map((item) => (
//                 <a
//                   key={item.title}
//                   href={item.url}
//                   className="px-3 py-2 text-gray-300 hover:text-white text-sm transition-colors duration-200"
//                 >
//                   {item.title}
//                 </a>
//               ))}
//             </div>
//           </div>
//         </div>
//       </nav>
//       <main>{children}</main>
//     </div>
//   );
// }

import React from "react";
import { Link, useLocation } from "react-router-dom";

const navigationItems = [
];

export default function Layout({ children, currentPageName }) {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-white">
      <main>{children}</main>
    </div>
  );
}