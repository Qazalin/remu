
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_4n1>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b64 s[6:7], s[0:1], 0x10                            // 000000001704: F4040180 F8000010
	s_load_b128 s[0:3], s[0:1], null                           // 00000000170C: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001714: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 000000001718: 86059F0F
	v_mov_b32_e32 v1, 0                                        // 00000000171C: 7E020280
	s_lshl_b64 s[8:9], s[4:5], 2                               // 000000001720: 84888204
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s10, s6, s8                                      // 000000001728: 800A0806
	s_addc_u32 s11, s7, s9                                     // 00000000172C: 820B0907
	s_load_b128 s[4:7], s[2:3], null                           // 000000001730: F4080101 F8000000
	s_clause 0x3                                               // 000000001738: BF850003
	s_load_b32 s2, s[10:11], null                              // 00000000173C: F4000085 F8000000
	s_load_b32 s3, s[10:11], 0x10                              // 000000001744: F40000C5 F8000010
	s_load_b32 s12, s[10:11], 0x20                             // 00000000174C: F4000305 F8000020
	s_load_b32 s10, s[10:11], 0x30                             // 000000001754: F4000285 F8000030
	s_add_u32 s0, s0, s8                                       // 00000000175C: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001760: 82010901
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	v_fma_f32 v0, s4, s2, 0                                    // 000000001768: D6130000 02000404
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001770: BF870091
	v_fmac_f32_e64 v0, s5, s3                                  // 000000001774: D52B0000 00000605
	v_fmac_f32_e64 v0, s6, s12                                 // 00000000177C: D52B0000 00001806
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001784: BF870001
	v_fmac_f32_e64 v0, s7, s10                                 // 000000001788: D52B0000 00001407
	global_store_b32 v1, v0, s[0:1]                            // 000000001790: DC6A0000 00000001
	s_nop 0                                                    // 000000001798: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000179C: BFB60003
	s_endpgm                                                   // 0000000017A0: BFB00000
