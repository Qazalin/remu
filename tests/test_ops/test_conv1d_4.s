
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_10_3_2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	s_mul_i32 s8, s2, 6                                        // 00000000171C: 96088602
	s_lshl_b64 s[12:13], s[14:15], 2                           // 000000001720: 848C820E
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s12                                      // 000000001728: 80060C06
	s_addc_u32 s7, s7, s13                                     // 00000000172C: 82070D07
	s_ashr_i32 s9, s8, 31                                      // 000000001730: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001738: 84888208
	s_add_u32 s0, s0, s8                                       // 00000000173C: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001740: 82010901
	s_load_b128 s[8:11], s[0:1], null                          // 000000001744: F4080200 F8000000
	s_clause 0x2                                               // 00000000174C: BF850002
	s_load_b64 s[14:15], s[6:7], null                          // 000000001750: F4040383 F8000000
	s_load_b64 s[16:17], s[6:7], 0x2c                          // 000000001758: F4040403 F800002C
	s_load_b64 s[6:7], s[6:7], 0x58                            // 000000001760: F4040183 F8000058
	s_load_b64 s[0:1], s[0:1], 0x10                            // 000000001768: F4040000 F8000010
	s_waitcnt lgkmcnt(0)                                       // 000000001770: BF89FC07
	v_fma_f32 v0, s14, s8, 0                                   // 000000001774: D6130000 0200100E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000177C: BF870091
	v_fmac_f32_e64 v0, s15, s9                                 // 000000001780: D52B0000 0000120F
	v_fmac_f32_e64 v0, s16, s10                                // 000000001788: D52B0000 00001410
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001790: BF870091
	v_fmac_f32_e64 v0, s17, s11                                // 000000001794: D52B0000 00001611
	v_fmac_f32_e64 v0, s6, s0                                  // 00000000179C: D52B0000 00000006
	s_mul_i32 s0, s2, 10                                       // 0000000017A4: 96008A02
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 0000000017A8: BF870141
	v_fmac_f32_e64 v0, s7, s1                                  // 0000000017AC: D52B0000 00000207
	s_ashr_i32 s1, s0, 31                                      // 0000000017B4: 86019F00
	v_mov_b32_e32 v1, 0                                        // 0000000017B8: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000017BC: 84808200
	v_max_f32_e32 v0, 0, v0                                    // 0000000017C0: 20000080
	s_add_u32 s0, s4, s0                                       // 0000000017C4: 80000004
	s_addc_u32 s1, s5, s1                                      // 0000000017C8: 82010105
	s_add_u32 s0, s0, s12                                      // 0000000017CC: 80000C00
	s_addc_u32 s1, s1, s13                                     // 0000000017D0: 82010D01
	global_store_b32 v1, v0, s[0:1]                            // 0000000017D4: DC6A0000 00000001
	s_nop 0                                                    // 0000000017DC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017E0: BFB60003
	s_endpgm                                                   // 0000000017E4: BFB00000
