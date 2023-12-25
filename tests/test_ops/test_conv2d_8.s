
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_6_63_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	s_mul_i32 s8, s2, 9                                        // 00000000171C: 96088902
	s_lshl_b64 s[16:17], s[14:15], 2                           // 000000001720: 8490820E
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s16                                      // 000000001728: 80061006
	s_addc_u32 s7, s7, s17                                     // 00000000172C: 82071107
	s_ashr_i32 s9, s8, 31                                      // 000000001730: 86099F08
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001734: BF870499
	s_lshl_b64 s[8:9], s[8:9], 2                               // 000000001738: 84888208
	s_add_u32 s0, s0, s8                                       // 00000000173C: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001740: 82010901
	s_load_b256 s[8:15], s[0:1], null                          // 000000001744: F40C0200 F8000000
	s_clause 0x7                                               // 00000000174C: BF850007
	s_load_b32 s3, s[6:7], null                                // 000000001750: F40000C3 F8000000
	s_load_b32 s18, s[6:7], 0x1c                               // 000000001758: F4000483 F800001C
	s_load_b32 s19, s[6:7], 0x38                               // 000000001760: F40004C3 F8000038
	s_load_b32 s20, s[6:7], 0x134                              // 000000001768: F4000503 F8000134
	s_load_b32 s21, s[6:7], 0x150                              // 000000001770: F4000543 F8000150
	s_load_b32 s22, s[6:7], 0x16c                              // 000000001778: F4000583 F800016C
	s_load_b32 s23, s[6:7], 0x268                              // 000000001780: F40005C3 F8000268
	s_load_b32 s24, s[6:7], 0x284                              // 000000001788: F4000603 F8000284
	s_waitcnt lgkmcnt(0)                                       // 000000001790: BF89FC07
	v_fma_f32 v0, s3, s8, 0                                    // 000000001794: D6130000 02001003
	s_load_b32 s3, s[6:7], 0x2a0                               // 00000000179C: F40000C3 F80002A0
	s_load_b32 s1, s[0:1], 0x20                                // 0000000017A4: F4000040 F8000020
	s_mul_i32 s0, s2, 63                                       // 0000000017AC: 9600BF02
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017B0: BF870091
	v_fmac_f32_e64 v0, s18, s9                                 // 0000000017B4: D52B0000 00001212
	v_fmac_f32_e64 v0, s19, s10                                // 0000000017BC: D52B0000 00001413
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017C4: BF870091
	v_fmac_f32_e64 v0, s20, s11                                // 0000000017C8: D52B0000 00001614
	v_fmac_f32_e64 v0, s21, s12                                // 0000000017D0: D52B0000 00001815
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017D8: BF870091
	v_fmac_f32_e64 v0, s22, s13                                // 0000000017DC: D52B0000 00001A16
	v_fmac_f32_e64 v0, s23, s14                                // 0000000017E4: D52B0000 00001C17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017EC: BF8700A1
	v_fmac_f32_e64 v0, s24, s15                                // 0000000017F0: D52B0000 00001E18
	s_waitcnt lgkmcnt(0)                                       // 0000000017F8: BF89FC07
	v_fmac_f32_e64 v0, s3, s1                                  // 0000000017FC: D52B0000 00000203
	s_ashr_i32 s1, s0, 31                                      // 000000001804: 86019F00
	v_mov_b32_e32 v1, 0                                        // 000000001808: 7E020280
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000180C: 84808200
	s_delay_alu instid0(VALU_DEP_2)                            // 000000001810: BF870002
	v_max_f32_e32 v0, 0, v0                                    // 000000001814: 20000080
	s_add_u32 s0, s4, s0                                       // 000000001818: 80000004
	s_addc_u32 s1, s5, s1                                      // 00000000181C: 82010105
	s_add_u32 s0, s0, s16                                      // 000000001820: 80001000
	s_addc_u32 s1, s1, s17                                     // 000000001824: 82011101
	global_store_b32 v1, v0, s[0:1]                            // 000000001828: DC6A0000 00000001
	s_nop 0                                                    // 000000001830: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001834: BFB60003
	s_endpgm                                                   // 000000001838: BFB00000
