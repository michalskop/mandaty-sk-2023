import{_ as u,o as c,c as d,a as t,b as l,w as j,e as i,d as k,F as g,r as w,m as h,t as f,p as m,f as v,q as x}from"./entry-6bb65338.mjs";const D={},z={class:"navbar navbar-dark bg-primary bg-dark"},M={class:"container-fluid"},I=t("a",{class:"navbar-brand h1",href:"/"},[i("Mand\xE1ty.sk"),t("small",{class:"d-none d-sm-inline"}," v\xE1\u0161 prst na tepe doby")],-1),N={class:"dropdown"},q=t("button",{class:"btn btn-primary dropdown-toggle",type:"button",id:"dropdownMenu","data-bs-toggle":"dropdown","aria-expanded":"false"}," ... ",-1),B={class:"dropdown-menu dropdown-menu-end","aria-labelledby":"dropdownMenu"},C={class:"dropdown-item",type:"button"},S=i("\u{1F3DB}\uFE0F "),V=i("N\xE1rodn\xE1 rada 2020-2023"),T=t("li",null,[t("hr")],-1),E=t("li",null,[t("button",{class:"dropdown-item",type:"button"},[i("\u{1F1E8}\u{1F1FF} "),t("a",{href:"https://mandaty.cz/"},"Mand\xE1ty.cz")])],-1);function F(e,n){const a=k;return c(),d("div",null,[t("header",z,[t("div",M,[I,t("div",N,[q,t("ul",B,[t("li",null,[t("button",C,[S,l(a,{to:"/"},{default:j(()=>[V]),_:1})])]),T,E])])])])])}var H=u(D,[["render",F]]);const K={data:function(){return{supporters:[],url_darujme:"https://projects.kohovolit.eu/api/?projectId=1200738"}},mounted(){var e=this;fetch(this.url_darujme).then(function(n){return n.json()}).then(function(n){e.supporters=n.reverse().filter(function(a){return a.last})})},methods:{diffDays:function(e,n){return Math.ceil(Math.abs(e-n)/(1e3*3600*24))},bgClass:function(e){var n=Date.parse(e),a=new Date,s=a.getTime();return this.diffDays(n,s)>540?"bg-light":this.diffDays(n,s)>270?"bg-secondary":"bg-warning"},textClass:function(e){var n=Date.parse(e),a=new Date,s=a.getTime();return this.diffDays(n,s)>540?"text-secondary":this.diffDays(n,s)>270?"text-light":"text-dark"}}},b=e=>(m("data-v-3e835464"),e=e(),v(),e),L={class:""},O={class:"container alert alert-success"},A=b(()=>t("h3",{class:"p-4"},[i("\u2764\uFE0F "),t("small",null,"Tento projekt vznikol len v\u010Faka podpore:")],-1)),G={class:"d-flex flex-row flex-wrap justify-content-around"},J=b(()=>t("div",{class:"mt-5"},[t("a",{href:"https://www.darujme.cz/projekt/1200738",target:"_blank"},[t("h4",{class:"outlink"},"Bu\u010Fte ako oni, podporte \u010Fal\u0161\xED rozvoj Mand\xE1ty.sk")])],-1));function P(e,n,a,s,r,o){return c(),d("div",L,[t("div",O,[A,t("div",G,[(c(!0),d(g,null,w(e.supporters,(_,p)=>(c(),d("div",{key:p,class:h(["card p-2 m-2",o.bgClass(_.date)])},[t("h6",{class:h(o.textClass(_.date))},f(_.given_name)+" "+f(_.family_name),3)],2))),128))]),J])])}var Q=u(K,[["render",P],["__scopeId","data-v-3e835464"]]);const R={mounted(){+function(e,n,a,s,r,o){e.DarujmeObject=s,e[s]=e[s]||function(){(e[s].q=e[s].q||[]).push(arguments)},r=n.createElement(a),o=n.getElementsByTagName(a)[0],r.async=1,r.src="https://www.darujme.cz/assets/scripts/widget.js",o.parentNode.insertBefore(r,o)}(window,document,"script","Darujme"),Darujme(1,"e2esjcadvq7fynj6","render","https://www.darujme.cz/widget?token=e2esjcadvq7fynj6","100%")}},U=e=>(m("data-v-4a5580ae"),e=e(),v(),e),W={class:"container"},X=U(()=>t("div",{"data-darujme-widget-token":"e2esjcadvq7fynj6"},"\xA0",-1)),Y=[X];function Z(e,n,a,s,r,o){return c(),d("div",W,Y)}var tt=u(R,[["render",Z],["__scopeId","data-v-4a5580ae"]]);const et={},nt={class:"navbar navbar-light bg-light"},st=t("div",{class:"container d-flex justify-content-between"},[t("span",null,[i("Michal \u0160kop, KohoVolit.eu, autor "),t("a",{href:"https://volebnikalkulacka.cz",class:"m-1"},"Volebn\xED kalkula\u010Dky")]),t("span",null,[t("a",{href:"https://projects.kohovolit.eu"},"Dal\u0161\xED projekty autora"),i(" \u2022 Kontakt: "),t("a",{href:"http://kohovolit.eu/kontakt"},"KohoVolit.eu")])],-1),at=[st];function ot(e,n){return c(),d("footer",nt,at)}var rt=u(et,[["render",ot]]);const ct={head(){return{script:[{src:"https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"}],link:[{rel:"stylesheet",href:"https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/united/bootstrap.min.css"}]}}},dt=t("hr",null,null,-1);function it(e,n,a,s,r,o){const _=H,p=Q,y=tt,$=rt;return c(),d("div",null,[l(_),x(e.$slots,"default"),dt,l(p),l(y),l($)])}var lt=u(ct,[["render",it]]);export{lt as default};
