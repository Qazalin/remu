
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_64_64>:
	s_load_b128 s[4:7], s[0:1], null                           // 000000001700: F4080100 F8000000
	s_lshl_b32 s2, s15, 6                                      // 000000001708: 8402860F
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_add_i32 s2, s2, s14                                      // 000000001714: 81020E02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001718: BF870499
	s_ashr_i32 s3, s2, 31                                      // 00000000171C: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001720: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001728: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000172C: 82070307
	s_load_b32 s6, s[6:7], null                                // 000000001730: F4000183 F8000000
	s_add_i32 s7, s14, -2                                      // 000000001738: 8107C20E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000173C: BF8704B9
	s_cmp_gt_u32 s7, 59                                        // 000000001740: BF08BB07
	s_cselect_b32 s7, -1, 0                                    // 000000001744: 980780C1
	s_add_i32 s8, s15, -2                                      // 000000001748: 8108C20F
	s_cmp_gt_u32 s8, 59                                        // 00000000174C: BF08BB08
	s_cselect_b32 s8, -1, 0                                    // 000000001750: 980880C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001754: BF870499
	s_or_b32 s7, s7, s8                                        // 000000001758: 8C070807
	s_and_b32 vcc_lo, exec_lo, s7                              // 00000000175C: 8B6A077E
	s_mov_b32 s7, 0                                            // 000000001760: BE870080
	s_cbranch_vccnz 12                                         // 000000001764: BFA4000C <E_64_64+0x98>
	s_mul_i32 s8, s15, 60                                      // 000000001768: 9608BC0F
	s_mov_b32 s9, 0                                            // 00000000176C: BE890080
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 000000001770: BF8704D9
	s_lshl_b64 s[10:11], s[8:9], 2                             // 000000001774: 848A8208
	s_mov_b32 s15, s9                                          // 000000001778: BE8F0009
	s_add_u32 s7, s0, s10                                      // 00000000177C: 80070A00
	s_addc_u32 s8, s1, s11                                     // 000000001780: 82080B01
	s_lshl_b64 s[0:1], s[14:15], 2                             // 000000001784: 8480820E
	s_add_u32 s0, s7, s0                                       // 000000001788: 80000007
	s_addc_u32 s1, s8, s1                                      // 00000000178C: 82010108
	s_load_b32 s7, s[0:1], -0x1e8                              // 000000001790: F40001C0 F81FFE18
	s_waitcnt lgkmcnt(0)                                       // 000000001798: BF89FC07
	v_add_f32_e64 v0, s6, s7                                   // 00000000179C: D5030000 00000E06
	v_mov_b32_e32 v1, 0                                        // 0000000017A4: 7E020280
	s_add_u32 s0, s4, s2                                       // 0000000017A8: 80000204
	s_addc_u32 s1, s5, s3                                      // 0000000017AC: 82010305
	global_store_b32 v1, v0, s[0:1]                            // 0000000017B0: DC6A0000 00000001
	s_nop 0                                                    // 0000000017B8: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000017BC: BFB60003
	s_endpgm                                                   // 0000000017C0: BFB00000
