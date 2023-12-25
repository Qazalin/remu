
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_77_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	s_mul_i32 s10, s2, 3                                       // 00000000171C: 960A8302
	s_lshl_b64 s[8:9], s[14:15], 2                             // 000000001720: 8488820E
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s8                                       // 000000001728: 80060806
	s_addc_u32 s7, s7, s9                                      // 00000000172C: 82070907
	s_ashr_i32 s11, s10, 31                                    // 000000001730: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[10:11], s[10:11], 2                           // 000000001738: 848A820A
	s_add_u32 s0, s0, s10                                      // 00000000173C: 80000A00
	s_addc_u32 s1, s1, s11                                     // 000000001740: 82010B01
	s_load_b64 s[10:11], s[0:1], null                          // 000000001744: F4040280 F8000000
	s_clause 0x2                                               // 00000000174C: BF850002
	s_load_b32 s3, s[6:7], null                                // 000000001750: F40000C3 F8000000
	s_load_b32 s12, s[6:7], 0x134                              // 000000001758: F4000303 F8000134
	s_load_b32 s6, s[6:7], 0x268                               // 000000001760: F4000183 F8000268
	s_load_b32 s1, s[0:1], 0x8                                 // 000000001768: F4000040 F8000008
	s_mul_i32 s0, s2, 0x4d                                     // 000000001770: 9600FF02 0000004D
	s_waitcnt lgkmcnt(0)                                       // 000000001778: BF89FC07
	v_fma_f32 v0, s3, s10, 0                                   // 00000000177C: D6130000 02001403
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001784: BF870091
	v_fmac_f32_e64 v0, s12, s11                                // 000000001788: D52B0000 0000160C
	v_fmac_f32_e64 v0, s6, s1                                  // 000000001790: D52B0000 00000206
	s_ashr_i32 s1, s0, 31                                      // 000000001798: 86019F00
	v_mov_b32_e32 v1, 0                                        // 00000000179C: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017A0: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 0000000017A4: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 0000000017A8: 20000080
	s_add_u32 s0, s4, s0                                       // 0000000017AC: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017B0: 82010105
	s_add_u32 s0, s0, s8                                       // 0000000017B4: 80000800
	s_addc_u32 s1, s1, s9                                      // 0000000017B8: 82010901
	global_store_b32 v1, v0, s[0:1]                            // 0000000017BC: DC6A0000 00000001
	s_nop 0                                                    // 0000000017C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017C8: BFB60003
	s_endpgm                                                   // 0000000017CC: BFB00000
